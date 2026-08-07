import SwiftUI

/// Settings › System: the iOS twin of the two web tabs that own everything this
/// screen edits — Settings → Recording (detection model, default detection mode,
/// confidence: frontend/src/pages/settings/RecordingTab.tsx) and Settings →
/// System (public URL, WebRTC candidates: SystemTab.tsx) — so an admin can
/// change detection + addresses without opening the web app.
///
/// **Every write here is PATCH /api/settings — NEVER PUT.** PUT is a
/// full-document replace and every field carries a backend default, so any key
/// the body omits is RESET rather than left alone: a PUT missing
/// `notifications.apns.direct.p8` destroys the APNs signing key and silently
/// breaks push (verified empirically). PATCH deep-merges, so each Save below
/// sends ONLY its own subtree (`detection` or `system`) and everything else is
/// preserved untouched. That is also why we decode the deliberately-minimal
/// `SettingsDocument` rather than modelling the whole document: what the app
/// can't model, it can't clobber.
///
/// Admin-only: reached from `SettingsHomeView`'s `session.isAdmin` section, so
/// it lives inside that view's existing NavigationStack — no NavigationStack of
/// its own.
struct SystemSettingsView: View {
    @EnvironmentObject private var session: SessionModel

    /// Last document loaded from the server — the baseline the drafts diff
    /// against to decide whether a Save is enabled.
    @State private var doc: SettingsDocument?
    @State private var loading = true
    @State private var loadError: String?

    // Detection draft
    @State private var modelKey = ""
    @State private var confidence = 0.5
    @State private var defaultMode: DetectMode = .always
    @State private var backend: DetectionBackend = .auto
    @State private var coralModel = CoralModelInfo.defaultKey
    @State private var savingDetection = false
    @State private var detectionError: String?

    // Addresses draft
    @State private var publicUrl = ""
    @State private var candidates: [String] = []
    @State private var newCandidate = ""
    @State private var savingSystem = false
    @State private var systemError: String?

    // Server draft — the nightly restart schedule is part of the `system`
    // subtree, so it rides the same Save button as the addresses (see
    // `saveSystem`) rather than a PATCH of its own that could clobber the public
    // URL / WebRTC list.
    @State private var autoRestartEnabled = false
    @State private var autoRestartTime = SystemSettingsView.date(fromRestartTime: SystemSettingsView.defaultRestartTime)

    // Restart-now action — an immediate POST, NOT a saved setting, so it stays
    // out of the system dirty/save flow entirely.
    @State private var showRestartConfirm = false
    @State private var restarting = false
    @State private var restartError: String?

    /// Model tiers offered by the backend (GET /api/detection/models). Empty
    /// when that call fails — the picker then degrades to the active key alone
    /// rather than blocking the rest of the screen.
    @State private var models: [DetectionModel] = []
    @State private var modelsFailed = false

    // Model download/delete draft — the web model-manager, brought into the app.
    // Separate from the detection Save flow: downloading a tier only FETCHES its
    // weights, it does NOT activate it (that stays Save detection). `downloadingKeys`
    // bridges the gap between the 202 POST and the first poll that reports the
    // "downloading" state; `pollingModels` guards against overlapping re-poll
    // loops when several Download buttons are tapped in a row.
    @State private var downloadingKeys: Set<String> = []
    @State private var pollingModels = false
    @State private var modelMgmtError: String?

    // Camera time-sync draft — the `time_sync` subtree, saved on its own button
    // (a time_sync-only PATCH deep-merges, so it can't clobber detection/system).
    // The `*Baseline` fields capture what `apply` seeded so Save enables only
    // after a real edit — needed because an empty stored zone is seeded to this
    // device's own zone, which would otherwise read as dirty on load.
    @State private var timeSyncAuto = false
    @State private var timeSyncZone = ""
    @State private var timeSyncAutoBaseline = false
    @State private var timeSyncZoneBaseline = ""
    @State private var savingTimeSync = false
    @State private var timeSyncError: String?

    /// The backend clamps confidence to 0.2…0.9 (DetectionSettings in
    /// backend/app/routers/settings.py) — match it so the slider can't produce a
    /// 422.
    private static let confidenceRange = 0.2 ... 0.9

    /// Timezone Picker options — every IANA id, sorted once. Safe at static
    /// scope: `knownTimeZoneIdentifiers` is a fixed list and reads no wall clock
    /// (unlike `Date()` / `TimeZone.current`, which must stay inside the body).
    private static let knownTimezones = TimeZone.knownTimeZoneIdentifiers.sorted()

    var body: some View {
        Group {
            if let loadError, doc == nil {
                ContentUnavailableView(
                    "Couldn't load settings",
                    systemImage: "gearshape.2",
                    description: Text(loadError)
                )
            } else {
                form
            }
        }
        .background(Theme.bg)
        .navigationTitle("System")
        .navigationBarTitleDisplayMode(.inline)
        .task { await load() }
        .refreshable { await load() }
    }

    private var form: some View {
        List {
            detectionHardwareSection
            detectionModelSection
            modelManagementSection
            confidenceSection
            defaultModeSection
            detectionSaveSection
            publicUrlSection
            candidatesSection
            autoRestartSection
            systemSaveSection
            timeSyncSection
            serverRestartSection
        }
        .scrollContentBackground(.hidden)
        .overlay {
            if loading && doc == nil {
                ProgressView().tint(Theme.accent)
            }
        }
        .disabled(doc == nil)
    }

    // MARK: - Detection › model

    /// Picker options: the backend's tiers, plus the currently-active key if the
    /// models call failed or doesn't list it — never silently drop the value
    /// that's actually configured.
    private var modelOptions: [String] {
        var keys = models.map(\.key)
        if !modelKey.isEmpty && !keys.contains(modelKey) {
            keys.insert(modelKey, at: 0)
        }
        return keys
    }

    /// Tier label ("Balanced"), falling back to the raw key for a model the
    /// backend didn't describe.
    private func modelLabel(_ key: String) -> String {
        models.first { $0.key == key }?.label ?? key
    }

    /// Which silicon runs inference. Mirrors the web segmented control.
    /// Applied at BOOT, so a change here needs a backend restart — unlike
    /// model/confidence, which reconfigure the live detector.
    private var detectionHardwareSection: some View {
        Section {
            Picker("Hardware", selection: $backend) {
                ForEach(DetectionBackend.allCases, id: \.self) { b in
                    Text(b.label).tag(b)
                }
            }
            .pickerStyle(.segmented)

            Text(backend.blurb)
                .font(.caption)
                .foregroundStyle(Theme.textSecondary)

            if backend == .coral {
                Text("Requires a Coral Edge TPU fitted to the server. If it is missing, "
                     + "detection will not start at all. Accuracy also drops "
                     + "(COCO mAP ~54 → ~33), hardest on small, distant and night-time "
                     + "people.")
                    .font(.caption2)
                    .foregroundStyle(Theme.warning)
            }
        } header: {
            Text("Detection hardware")
        } footer: {
            Text("Takes effect after a backend restart (Settings → System → Restart server).")
        }
        .listRowBackground(Theme.surface)
    }

    /// Edge TPU model list, shown INSTEAD of the D-FINE tiers when Coral is
    /// selected — the two lists are disjoint, so showing both would invite an
    /// invalid pairing.
    private var coralModelSection: some View {
        Section {
            Picker("Model", selection: $coralModel) {
                ForEach(CoralModelInfo.all) { m in
                    Text(m.label).tag(m.key)
                }
            }
            .tint(Theme.accent)

            // Plain Strings, NOT interpolated Text: SwiftUI's `specifier:` form
            // yields a LocalizedStringKey, which cannot be concatenated with +.
            let info = CoralModelInfo.find(coralModel)
            // Same shape as the GPU model row below: blurb, then meta, then any
            // warning — so the Edge TPU does not read as the lesser backend.
            let meta = String(
                format: "%@ · %.1f mAP · ~%.1f ms · %dpx",
                CoralModelInfo.vocabulary, info.map, info.latencyMs, info.inputSize
            ) + (info.note.isEmpty ? "" : " · \(info.note)")
            let slowWarning = String(
                format: "At ~%.0f ms this sustains under 10 inferences/sec — roughly what "
                    + "two cameras at 5 fps already demand. Frames will be dropped under load.",
                info.latencyMs
            )
            VStack(alignment: .leading, spacing: 4) {
                Text(info.blurb)
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
                Text(meta)
                    .font(.caption2)
                    .foregroundStyle(Theme.textSecondary)
                if info.slow {
                    Text(slowWarning)
                        .font(.caption2)
                        .foregroundStyle(Theme.warning)
                }
            }
        } header: {
            Text("Edge TPU model")
        } footer: {
            Text("Downloaded and checksum-verified on first use. Switching models reloads "
                 + "the detector — detection pauses for a few seconds.")
        }
        .listRowBackground(Theme.surface)
    }

    @ViewBuilder
    private var detectionModelSection: some View {
        if backend == .coral {
            coralModelSection
        } else {
        Section {
            Picker("Model", selection: $modelKey) {
                ForEach(modelOptions, id: \.self) { key in
                    modelRow(key).tag(key)
                }
            }
            .tint(Theme.accent)

            if let info = models.first(where: { $0.key == modelKey }) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(info.blurb)
                        .font(.caption)
                        .foregroundStyle(Theme.textSecondary)
                    Text(modelMeta(info))
                        .font(.caption2)
                        .foregroundStyle(Theme.textSecondary)
                }
            }
        } header: {
            Text("Detection model")
        } footer: {
            if modelsFailed {
                Text("Couldn't load the model list from the server — showing the model that's currently configured. Manage tiers (download, delete) in the web app.")
            } else {
                Text("Pick a tier to match your hardware. Switching tiers causes a brief detection gap while the new model loads. A tier that isn't downloaded yet starts downloading when you save; you can also download or remove tiers under Manage models below.")
            }
        }
        .listRowBackground(Theme.surface)
        }
    }

    /// One picker row: the tier label plus its live state — which model the
    /// detector is running now, and which are downloaded vs not yet on disk.
    private func modelRow(_ key: String) -> some View {
        let info = models.first { $0.key == key }
        return HStack(spacing: 6) {
            Text(modelLabel(key))
            if info?.active == true {
                statePill("Active", Theme.success)
            } else {
                switch info?.state {
                case "ready":
                    statePill("Downloaded", Theme.textSecondary)
                case "downloading", "verifying":
                    statePill("Downloading", Theme.warning)
                case "error":
                    statePill("Failed", Theme.danger)
                case "absent":
                    statePill("Not downloaded", Theme.warning)
                default:
                    EmptyView()
                }
            }
        }
    }

    private func statePill(_ text: String, _ color: Color) -> some View {
        Text(text)
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color.opacity(0.18))
            .foregroundStyle(color)
            .clipShape(Capsule())
    }

    private func modelMeta(_ info: DetectionModel) -> String {
        var parts: [String] = []
        if info.numClasses > 0 && !info.vocabulary.isEmpty {
            parts.append("\(info.vocabulary) · \(info.numClasses) classes")
        }
        parts.append("\(Int((Double(info.sizeBytes) / 1_000_000).rounded())) MB")
        parts.append(String(format: "%.1f mAP", info.approxMap))
        parts.append("\(info.inputSize)px")
        return parts.joined(separator: " · ")
    }

    // MARK: - Detection › model management (download / delete)

    /// Per-tier download / delete — the web model manager, brought into the app.
    /// The Picker above stays the ACTIVATE control (select a tier + Save
    /// detection to run it); this section only fetches a tier's weights when it
    /// isn't on disk yet, and frees the ones you don't use. Coral tiers aren't
    /// listed — they're fetched on first use, not through these endpoints — so it
    /// only appears for the GPU/D-FINE list, and only once that list has loaded.
    @ViewBuilder
    private var modelManagementSection: some View {
        if backend != .coral && !models.isEmpty {
            Section {
                ForEach(models) { model in
                    HStack {
                        modelRow(model.key)
                        Spacer()
                        modelManagementControl(model)
                    }
                    .swipeActions(edge: .trailing, allowsFullSwipe: false) {
                        // Only a downloaded, non-active tier can go: the backend
                        // 409s on the active one (handled in `deleteModel`).
                        if canDelete(model) {
                            Button(role: .destructive) {
                                Task { await deleteModel(model.key) }
                            } label: {
                                Label("Delete", systemImage: "trash")
                            }
                        }
                    }
                }

                if let modelMgmtError {
                    Text(modelMgmtError)
                        .font(.footnote)
                        .foregroundStyle(Theme.danger)
                }
            } header: {
                Text("Manage models")
            } footer: {
                Text("Download a tier to keep its weights ready on disk, or swipe a downloaded one to delete it. The active tier can't be deleted — switch tiers first. A download keeps running on the server even if you leave this screen.")
            }
            .listRowBackground(Theme.surface)
        }
    }

    /// Trailing control for a management row: a Download/Retry button when the
    /// tier isn't on disk, or live progress while it is fetching. A downloaded
    /// tier shows nothing here — its pill already reads "Downloaded"/"Active" and
    /// delete is the swipe action.
    @ViewBuilder
    private func modelManagementControl(_ model: DetectionModel) -> some View {
        if downloadingKeys.contains(model.key)
            || model.state == "downloading" || model.state == "verifying" {
            HStack(spacing: 6) {
                if model.progressPct > 0 {
                    Text("\(Int(model.progressPct.rounded()))%")
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(Theme.textSecondary)
                }
                ProgressView().tint(Theme.accent)
            }
        } else if model.state == "absent" || model.state == "error" {
            // "error" is a failed/partial download — offer a retry, same button.
            Button(model.state == "error" ? "Retry" : "Download") {
                Task { await startModelDownload(model.key) }
            }
            .font(.caption.weight(.semibold))
            .buttonStyle(.borderless)
            .foregroundStyle(Theme.accent)
        }
    }

    /// A tier is removable only when it's downloaded and NOT the one running —
    /// the backend returns 409 for the active tier.
    private func canDelete(_ model: DetectionModel) -> Bool {
        model.state == "ready" && !model.active
    }

    /// POST the download (202), then re-poll so the pill + progress update live.
    private func startModelDownload(_ key: String) async {
        guard let api = session.api, !downloadingKeys.contains(key) else { return }
        downloadingKeys.insert(key)
        modelMgmtError = nil
        do {
            try await api.downloadModel(key: key)
        } catch {
            session.handleAPIError(error)
            modelMgmtError = (error as? ApiError)?.message ?? error.localizedDescription
            downloadingKeys.remove(key)
            return
        }
        await pollModels()
    }

    /// DELETE the tier's file, then refresh the list so its pill drops back to
    /// "Not downloaded". A 409 means it's the active tier — surface the fix
    /// through the section's own error line rather than a raw HTTP message.
    private func deleteModel(_ key: String) async {
        guard let api = session.api else { return }
        modelMgmtError = nil
        do {
            try await api.deleteModel(key: key)
        } catch {
            session.handleAPIError(error)
            if (error as? ApiError)?.status == 409 {
                modelMgmtError = "That tier is active — switch tier first (pick another model and Save detection), then delete it."
            } else {
                modelMgmtError = (error as? ApiError)?.message ?? error.localizedDescription
            }
            return
        }
        await loadModels()
    }

    /// Re-poll `detectionModels()` on a ~2s cadence so a just-triggered download
    /// updates its state pill / progress without a manual refresh. Guarded by
    /// `pollingModels` so a burst of Download taps shares ONE loop instead of
    /// stacking pollers that all write `models`. Stops once nothing is mid-
    /// download; capped so a wedged download can't poll forever (a pull-to-
    /// refresh or re-open resumes it).
    private func pollModels() async {
        guard !pollingModels else { return }
        pollingModels = true
        defer { pollingModels = false }
        for _ in 0 ..< 120 {
            try? await Task.sleep(nanoseconds: 2_000_000_000)
            guard let api = session.api else { return }
            guard let fresh = try? await api.detectionModels() else { continue }
            models = fresh.models
            // Drop the in-flight marker once the poll has caught up (state has
            // left "absent"); the pill drives the row from there.
            downloadingKeys = downloadingKeys.filter { key in
                fresh.models.first { $0.key == key }?.state == "absent"
            }
            let busy = !downloadingKeys.isEmpty || fresh.models.contains {
                $0.state == "downloading" || $0.state == "verifying"
            }
            if !busy { return }
        }
    }

    // MARK: - Detection › confidence

    private var confidenceSection: some View {
        Section {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("Threshold")
                        .foregroundStyle(Theme.textSecondary)
                    Spacer()
                    Text("\(Int((confidence * 100).rounded()))%")
                        .font(.subheadline.weight(.medium).monospacedDigit())
                        .foregroundStyle(Theme.textPrimary)
                }
                Slider(
                    value: $confidence,
                    in: Self.confidenceRange,
                    step: 0.05
                )
                .tint(Theme.accent)
                .accessibilityLabel("Confidence threshold")
                .accessibilityValue("\(Int((confidence * 100).rounded())) percent")
            }
            .padding(.vertical, 2)
        } header: {
            Text("Confidence")
        } footer: {
            Text("Lower catches more (and more false positives); higher only keeps sure detections.")
        }
        .listRowBackground(Theme.surface)
    }

    // MARK: - Detection › default mode

    private var defaultModeSection: some View {
        Section {
            Picker("Detection runs", selection: $defaultMode) {
                ForEach(DetectMode.allCases, id: \.self) { mode in
                    Text(modeLabel(mode)).tag(mode)
                }
            }
            .pickerStyle(.segmented)
            Text(modeExplanation(defaultMode))
                .font(.caption)
                .foregroundStyle(Theme.textSecondary)
        } header: {
            Text("Default detection mode")
        } footer: {
            Text("How the GPU detector is scheduled for a newly added camera. Cameras with their own on-camera AI (SMD human/vehicle, IVS tripwire/intrusion) can gate detection on that signal to cut GPU load. Change it per camera under Settings → Cameras.")
        }
        .listRowBackground(Theme.surface)
    }

    /// Segment titles mirror the web's advanced segmented control (RecordingTab).
    private func modeLabel(_ mode: DetectMode) -> String {
        switch mode {
        case .always: return "Server"
        case .cameraAi: return "Camera-triggered"
        case .cameraAiOnly: return "On-camera only"
        }
    }

    /// Per-mode copy taken from the web's control hints.
    private func modeExplanation(_ mode: DetectMode) -> String {
        switch mode {
        case .always:
            return "New cameras run continuous server detection — the safe default: detection can never silently stop."
        case .cameraAi:
            return "New cameras run server detection only while their own AI sees motion — big GPU savings; may miss what the camera AI misses. Cameras without on-camera AI keep detecting continuously."
        case .cameraAiOnly:
            return "New cameras rely on their on-camera AI alone — no server inference at all. Cameras without on-camera AI keep detecting continuously."
        }
    }

    // MARK: - Detection › save

    private var detectionDirty: Bool {
        guard let doc else { return false }
        return modelKey != doc.detection.model
            || abs(confidence - doc.detection.confidence) > 0.0001
            || defaultMode != doc.detection.defaultMode
    }

    private var detectionSaveSection: some View {
        Section {
            Button {
                Task { await saveDetection() }
            } label: {
                HStack {
                    Spacer()
                    if savingDetection {
                        ProgressView().tint(Theme.accent)
                    } else {
                        Text("Save detection")
                            .foregroundStyle(detectionDirty ? Theme.accent : Theme.textSecondary)
                    }
                    Spacer()
                }
            }
            .disabled(savingDetection || !detectionDirty || modelKey.isEmpty)

            if let detectionError {
                Text(detectionError)
                    .font(.footnote)
                    .foregroundStyle(Theme.danger)
            }
        } footer: {
            Text("Saves only the detection settings — nothing else on the server is touched. Changing the model reloads the detector: there's a short gap with no detection while the new one loads.")
        }
        .listRowBackground(Theme.surface)
    }

    private func saveDetection() async {
        guard let api = session.api, !savingDetection else { return }
        savingDetection = true
        defer { savingDetection = false }
        detectionError = nil
        // ONLY the detection subtree — the server deep-merges it and preserves
        // every other block (notifications/APNs key, recording, mqtt, time_sync).
        let patch = SettingsPatch(
            detection: .init(
                model: modelKey, confidence: confidence, defaultMode: defaultMode,
                backend: backend, coralModel: coralModel
            )
        )
        do {
            apply(try await api.patchSettings(patch))
        } catch {
            session.handleAPIError(error)
            detectionError = (error as? ApiError)?.message ?? error.localizedDescription
        }
    }

    // MARK: - Addresses › public URL

    private var publicUrlSection: some View {
        Section {
            TextField("https://nvr.example.com", text: $publicUrl)
                .keyboardType(.URL)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .foregroundStyle(Theme.textPrimary)
        } header: {
            Text("Public URL")
        } footer: {
            Text("The externally reachable base URL used in notification click-links, e.g. https://nvr.tailnet-name.ts.net. Leave empty for LAN-only use.")
        }
        .listRowBackground(Theme.surface)
    }

    // MARK: - Addresses › WebRTC candidates

    private var candidatesSection: some View {
        Section {
            // Index-keyed: candidate strings aren't guaranteed unique, so
            // `id: \.self` on the value could collide mid-edit.
            ForEach(candidates.indices, id: \.self) { index in
                Text(candidates[index])
                    .font(.subheadline.monospaced())
                    .foregroundStyle(Theme.textPrimary)
            }
            .onDelete { candidates.remove(atOffsets: $0) }

            HStack {
                TextField("192.168.1.10:8555", text: $newCandidate)
                    .keyboardType(.numbersAndPunctuation)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .onSubmit(addCandidate)
                Button(action: addCandidate) {
                    Image(systemName: "plus.circle.fill")
                        .foregroundStyle(canAddCandidate ? Theme.accent : Theme.textSecondary)
                }
                .buttonStyle(.plain)
                .disabled(!canAddCandidate)
                .accessibilityLabel("Add WebRTC address")
            }
        } header: {
            Text("WebRTC addresses")
        } footer: {
            Text("Extra addresses this server's live-view port can be reached on — add the server's LAN IP and (if used) its Tailscale IP, each as ip:8555. Leave empty to use the defaults; live view falls back to the slower MSE path when none are reachable. Swipe a row to remove it.")
        }
        .listRowBackground(Theme.surface)
    }

    /// The backend caps entries at 64 characters and the list at 16; gate the +
    /// on that plus "non-empty and not already listed" and let the server's
    /// readable 422 carry anything subtler.
    private var canAddCandidate: Bool {
        let value = newCandidate.trimmedWhitespace
        return !value.isEmpty
            && value.count <= 64
            && candidates.count < 16
            && !candidates.contains(value)
    }

    private func addCandidate() {
        guard canAddCandidate else { return }
        candidates.append(newCandidate.trimmedWhitespace)
        newCandidate = ""
    }

    // MARK: - Addresses › save

    private var systemDirty: Bool {
        guard let doc else { return false }
        return publicUrl.trimmedWhitespace != doc.system.publicUrl
            || candidates != doc.system.webrtcCandidates
            || autoRestartEnabled != doc.system.autoRestart.enabled
            || autoRestartTimeString != doc.system.autoRestart.time
    }

    private var systemSaveSection: some View {
        Section {
            Button {
                Task { await saveSystem() }
            } label: {
                HStack {
                    Spacer()
                    if savingSystem {
                        ProgressView().tint(Theme.accent)
                    } else {
                        Text("Save server settings")
                            .foregroundStyle(systemDirty ? Theme.accent : Theme.textSecondary)
                    }
                    Spacer()
                }
            }
            .disabled(savingSystem || !systemDirty)

            if let systemError {
                Text(systemError)
                    .font(.footnote)
                    .foregroundStyle(Theme.danger)
            }
        } footer: {
            Text("Saves the public URL, WebRTC addresses, and nightly restart schedule — nothing else on the server is touched. Changing the WebRTC addresses regenerates the streaming config, so live view reconnects.")
        }
        .listRowBackground(Theme.surface)
    }

    private func saveSystem() async {
        guard let api = session.api, !savingSystem else { return }
        savingSystem = true
        defer { savingSystem = false }
        systemError = nil
        // ONLY the system subtree — see saveDetection(). Public URL, WebRTC
        // candidates and the nightly restart schedule all live here, so ONE
        // PATCH carries all three; the backend deep-merges and leaves every
        // other block untouched. Bundling auto-restart in rather than giving it
        // its own PATCH is what keeps the URL/WebRTC list from being clobbered.
        let patch = SettingsPatch(
            system: .init(
                publicUrl: publicUrl.trimmedWhitespace,
                webrtcCandidates: candidates
                    .map(\.trimmedWhitespace)
                    .filter { !$0.isEmpty },
                autoRestart: .init(
                    enabled: autoRestartEnabled,
                    time: autoRestartTimeString
                )
            )
        )
        do {
            apply(try await api.patchSettings(patch))
        } catch {
            session.handleAPIError(error)
            systemError = (error as? ApiError)?.message ?? error.localizedDescription
        }
    }

    // MARK: - Camera time-sync

    /// Timezone Picker options — the known zones plus the stored one if the
    /// server holds a value the list doesn't (mirrors `modelOptions`), so the
    /// configured zone is never silently dropped.
    private var timezoneOptions: [String] {
        var zones = Self.knownTimezones
        if !timeSyncZone.isEmpty && !zones.contains(timeSyncZone) {
            zones.insert(timeSyncZone, at: 0)
        }
        return zones
    }

    private var timeSyncDirty: Bool {
        timeSyncAuto != timeSyncAutoBaseline || timeSyncZone != timeSyncZoneBaseline
    }

    /// Automatic camera-clock provisioning (`time_sync`) — its own Section +
    /// Save because it's a subtree of its own; a time_sync-only PATCH deep-merges
    /// and leaves detection/system untouched. When the server holds no zone the
    /// Picker defaults to this device's zone (seeded in `apply`, never at static
    /// scope — `TimeZone.current` reads a wall clock).
    private var timeSyncSection: some View {
        Section {
            Toggle("Auto-sync camera time", isOn: $timeSyncAuto)
                .tint(Theme.accent)

            // A pushed list, not a menu: there are a few hundred IANA zones.
            Picker("Timezone", selection: $timeSyncZone) {
                ForEach(timezoneOptions, id: \.self) { zone in
                    Text(zone).tag(zone)
                }
            }
            .pickerStyle(.navigationLink)
            .tint(Theme.accent)

            Button {
                Task { await saveTimeSync() }
            } label: {
                HStack {
                    Spacer()
                    if savingTimeSync {
                        ProgressView().tint(Theme.accent)
                    } else {
                        Text("Save time sync")
                            .foregroundStyle(timeSyncDirty ? Theme.accent : Theme.textSecondary)
                    }
                    Spacer()
                }
            }
            .disabled(savingTimeSync || !timeSyncDirty)

            if let timeSyncError {
                Text(timeSyncError)
                    .font(.footnote)
                    .foregroundStyle(Theme.danger)
            }
        } header: {
            Text("Camera time sync")
        } footer: {
            Text("Keeps each camera's clock set from the server so event and recording timestamps stay aligned. Pick the timezone the cameras are installed in. Saves only this setting — nothing else on the server is touched.")
        }
        .listRowBackground(Theme.surface)
    }

    private func saveTimeSync() async {
        guard let api = session.api, !savingTimeSync else { return }
        savingTimeSync = true
        defer { savingTimeSync = false }
        timeSyncError = nil
        // ONLY the time_sync subtree — the backend deep-merges it and leaves
        // every other block (detection, system, notifications/APNs key) alone.
        let patch = SettingsPatch(
            timeSync: .init(autoSync: timeSyncAuto, timezone: timeSyncZone)
        )
        do {
            apply(try await api.patchSettings(patch))
        } catch {
            session.handleAPIError(error)
            timeSyncError = (error as? ApiError)?.message ?? error.localizedDescription
        }
    }

    // MARK: - Server › nightly auto-restart

    /// Nightly self-restart schedule. It's part of the `system` subtree, so it
    /// is persisted by the Save button below (see `saveSystem`) — NOT a PATCH of
    /// its own, which would have to resend the URL/WebRTC list and could clobber
    /// a concurrent edit. Sits directly above that button so the grouping reads.
    private var autoRestartSection: some View {
        Section {
            Toggle("Nightly auto-restart", isOn: $autoRestartEnabled)
                .tint(Theme.accent)

            if autoRestartEnabled {
                DatePicker(
                    "Restart time",
                    selection: $autoRestartTime,
                    displayedComponents: .hourAndMinute
                )
                .tint(Theme.accent)
            }
        } header: {
            Text("Nightly restart")
        } footer: {
            Text("Restarts the server automatically at this local time every night — a cheap way to shed any slow leak while nobody's watching. Saved with the button below.")
        }
        .listRowBackground(Theme.surface)
    }

    // MARK: - Server › restart now

    /// Immediate self-restart (POST /api/system/restart, 202). A one-shot
    /// action, not a saved setting, so it stays out of the system PATCH flow.
    /// Confirmation-gated because it takes the API offline for ~15s.
    private var serverRestartSection: some View {
        Section {
            Button(role: .destructive) {
                showRestartConfirm = true
            } label: {
                HStack {
                    Spacer()
                    if restarting {
                        ProgressView().tint(Theme.accent)
                    } else {
                        Text("Restart server")
                            .foregroundStyle(Theme.danger)
                    }
                    Spacer()
                }
            }
            .disabled(restarting)
            .confirmationDialog(
                "Restart server?",
                isPresented: $showRestartConfirm,
                titleVisibility: .visible
            ) {
                Button("Restart server", role: .destructive) {
                    Task { await restartServer() }
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("Restarting takes the API offline for about 15 seconds while the server comes back. Live view and this app reconnect once it's up.")
            }

            if restarting {
                Text("Restarting… the server will be back in a few seconds.")
                    .font(.footnote)
                    .foregroundStyle(Theme.textSecondary)
            }

            if let restartError {
                Text(restartError)
                    .font(.footnote)
                    .foregroundStyle(Theme.danger)
            }
        } header: {
            Text("Server")
        } footer: {
            Text("Restarts the backend process now. Use after changing a setting that only takes effect at boot — for example the detection hardware above.")
        }
        .listRowBackground(Theme.surface)
    }

    private func restartServer() async {
        guard let api = session.api, !restarting else { return }
        restarting = true
        restartError = nil
        do {
            try await api.restartServer()
        } catch {
            // Surface it like the Save sections do and re-enable at once —
            // nothing is coming back down, so there's no outage to wait out.
            session.handleAPIError(error)
            restartError = (error as? ApiError)?.message ?? error.localizedDescription
            restarting = false
            return
        }
        // Hold the button disabled through the outage so a second tap can't hit
        // a half-down server. ~8s comfortably covers a normal restart inside the
        // ~15s window advertised above; reset by a sleep, not a health poll.
        try? await Task.sleep(nanoseconds: 8_000_000_000)
        restarting = false
    }

    // MARK: - Nightly restart time <-> "HH:mm"

    /// Fixed-locale, fixed-calendar formatter for `system.auto_restart.time`.
    /// en_US_POSIX + an explicit Gregorian calendar so a device on 12-hour
    /// display or a non-Gregorian calendar still round-trips the exact 24-hour
    /// "HH:mm" the backend stores. Built once; NEVER seeded from `Date()`.
    private static let restartTimeFormatter: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.calendar = Calendar(identifier: .gregorian)
        f.dateFormat = "HH:mm"
        return f
    }()

    /// Fallback for a missing or unparseable stored value — a string, not a
    /// `Date()`, so `date(fromRestartTime:)` is total and carries no wall-clock
    /// seed (which is also why the `autoRestartTime` @State default is safe).
    private static let defaultRestartTime = "04:00"

    /// "HH:mm" -> a `Date` carrying just that hour/minute for the picker.
    private static func date(fromRestartTime hhmm: String) -> Date {
        restartTimeFormatter.date(from: hhmm)
            ?? restartTimeFormatter.date(from: defaultRestartTime)!
    }

    /// The picker's `Date` -> the "HH:mm" string the backend stores.
    private var autoRestartTimeString: String {
        Self.restartTimeFormatter.string(from: autoRestartTime)
    }

    // MARK: - Loading

    private func load() async {
        guard let api = session.api else { return }
        loading = doc == nil
        do {
            apply(try await api.settingsDocument())
            loadError = nil
        } catch {
            session.handleAPIError(error)
            loadError = (error as? ApiError)?.message ?? error.localizedDescription
        }
        loading = false
        await loadModels()
    }

    /// The tier list is a nice-to-have: a failure here leaves the picker on the
    /// configured key rather than failing the screen.
    private func loadModels() async {
        guard let api = session.api else { return }
        do {
            models = try await api.detectionModels().models
            modelsFailed = false
        } catch {
            models = []
            modelsFailed = true
        }
    }

    /// Adopt a server document (initial load or a PATCH response) as both the
    /// baseline and the draft — the response is the full updated document, so a
    /// save leaves the form showing exactly what the server now holds
    /// (e.g. a public_url the backend stripped a trailing slash from).
    private func apply(_ document: SettingsDocument) {
        doc = document
        modelKey = document.detection.model
        confidence = document.detection.confidence
        defaultMode = document.detection.defaultMode
        backend = document.detection.backend
        coralModel = document.detection.coralModel
        publicUrl = document.system.publicUrl
        candidates = document.system.webrtcCandidates
        autoRestartEnabled = document.system.autoRestart.enabled
        autoRestartTime = Self.date(fromRestartTime: document.system.autoRestart.time)
        // Camera time-sync. An empty stored zone means "server default" — seed
        // the Picker to this device's zone so it shows something sensible.
        // `TimeZone.current` is read HERE, inside a method, never at static scope.
        timeSyncAuto = document.timeSync.autoSync
        let storedZone = document.timeSync.timezone
        let seededZone = storedZone.isEmpty ? TimeZone.current.identifier : storedZone
        timeSyncZone = seededZone
        // Baselines the Save diffs against, so seeding to the device zone above
        // isn't mistaken for an unsaved edit on load.
        timeSyncAutoBaseline = document.timeSync.autoSync
        timeSyncZoneBaseline = seededZone
    }
}

private extension String {
    var trimmedWhitespace: String { trimmingCharacters(in: .whitespacesAndNewlines) }
}
