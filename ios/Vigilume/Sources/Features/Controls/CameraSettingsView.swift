import SwiftUI

/// Full per-camera SETTINGS screen — iOS parity with the web surfaces
/// (frontend CameraDetail.tsx controls + CamerasTab.tsx edit form and device
/// modal). Everything here is ADMIN-ONLY: the entry point in CameraDetailView is
/// gated on `session.isAdmin`, and this view double-checks it. A viewer never
/// reaches it.
///
/// It groups the settings the way the backend routes them:
///   • Camera config (PUT /api/cameras/{name}, CameraUpdate) — friendly name,
///     model, IP, detect objects, detect mode, detect fps, RTSP main/sub
///     overrides and exempt zones. Saved together (blank creds = keep stored).
///   • Credentials (same PUT + POST /probe) — the camera's own username/password
///     with a Save & Test that surfaces the probe result.
///   • Device settings (PUT /api/cameras/{name}/settings, DeviceSettingsPatch)
///     — flip, OSD name, on-camera motion detect, privacy mode, mic/speaker
///     volume. (Night vision / spotlight / siren / reboot stay on the detail
///     Controls card.)
struct CameraSettingsView: View {
    let camera: Camera

    @EnvironmentObject private var session: SessionModel
    @Environment(\.dismiss) private var dismiss

    // MARK: Camera-config form state (seeded from the passed camera)
    @State private var friendlyName = ""
    @State private var model = ""
    @State private var ip = ""
    @State private var detectObjects: [String] = []
    @State private var detectMode: DetectMode = .always
    @State private var audioCodec: String = "g711a"
    @State private var smartSpotlight = false
    @State private var spotlightHoldSeconds: Int = 60
    @State private var detectFps: Double = 5
    @State private var mainUrl = ""
    @State private var subUrl = ""
    @State private var exemptZones: [ExemptZone] = []
    @State private var configSaving = false

    // MARK: Credentials
    @State private var credUser = ""
    @State private var credPass = ""
    @State private var credTesting = false
    @State private var probe: ProbeResult?

    // MARK: Device settings (Amcrest state behind the PUT /settings route)
    @State private var device: DeviceSettings?
    @State private var deviceLoading = false
    @State private var deviceError: String?
    @State private var flip = false
    @State private var osdName = ""
    @State private var motionDetect = false
    @State private var micVolume: Double = 0
    @State private var speakerVolume: Double = 0
    @State private var deviceSaving = false

    // MARK: Feedback
    @State private var banner: String?
    @State private var errorMessage: String?

    private static let knownModels = [
        "IP5M-T1277EW-AI", "IP8M-2779EW-AI", "AD410", "IP3M-941B", "IP4M-1041B", "IP4M-1056E",
    ]

    private var isOnline: Bool {
        session.cameraOnline[camera.name] ?? camera.online
    }

    var body: some View {
        Group {
            if session.isAdmin {
                content
            } else {
                // Defensive: a viewer must never see camera settings.
                ContentUnavailableCompat(
                    title: "Admin only",
                    systemImage: "lock.fill",
                    message: "Camera settings are available to administrators."
                )
            }
        }
        .background(Theme.bg.ignoresSafeArea())
        .navigationTitle("Settings")
        .navigationBarTitleDisplayMode(.inline)
        .task {
            seedFromCamera()
            await loadDeviceSettings()
        }
        .alert(
            "Camera settings",
            isPresented: Binding(
                get: { errorMessage != nil },
                set: { if !$0 { errorMessage = nil } }
            )
        ) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(errorMessage ?? "")
        }
    }

    private var content: some View {
        ScrollView {
            VStack(spacing: 16) {
                if let banner {
                    Label(banner, systemImage: "checkmark.circle.fill")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(Theme.success)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(10)
                        .background(
                            RoundedRectangle(cornerRadius: 10, style: .continuous)
                                .fill(Theme.success.opacity(0.12))
                        )
                }

                identityCard
                detectionCard
                streamsCard
                zonesCard
                credentialsCard
                deviceCard
            }
            .padding(16)
        }
    }

    // MARK: - Identity

    private var identityCard: some View {
        settingsCard("Identity", systemImage: "camera.fill") {
            labeledField("Friendly name") {
                TextField("Front Yard", text: $friendlyName)
                    .textFieldStyle(.roundedBorder)
                    .autocorrectionDisabled()
            }
            labeledField("Model") {
                Picker("Model", selection: $model) {
                    ForEach(modelOptions, id: \.self) { m in
                        Text(modelLabel(m)).tag(m)
                    }
                }
                .pickerStyle(.menu)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            labeledField("IP address") {
                TextField("192.168.1.101", text: $ip)
                    .textFieldStyle(.roundedBorder)
                    .keyboardType(.numbersAndPunctuation)
                    .autocorrectionDisabled()
                    .textInputAutocapitalization(.never)
            }
            saveButton("Save camera", busy: configSaving) {
                Task { await saveCameraConfig() }
            }
        }
    }

    /// Auto-detect first, then known models, then the camera's stored model
    /// (when non-standard, so a probe-adopted/legacy value is preserved).
    private var modelOptions: [String] {
        var opts = ["unknown"] + Self.knownModels
        if !model.isEmpty, model != "unknown", !opts.contains(model) {
            opts.append(model)
        }
        return opts
    }

    /// "unknown" is the backend's auto-detect sentinel: saving with it set
    /// makes the probe adopt whatever the device reports via getDeviceType.
    /// Show it as the action it performs, not as the raw sentinel.
    private func modelLabel(_ m: String) -> String {
        m == "unknown" ? "Auto-detect (recommended)" : m
    }

    // MARK: - Detection

    private var detectionCard: some View {
        settingsCard("Detection", systemImage: "sparkles") {
            labeledField("Detect objects") {
                ObjectPickerView(selected: $detectObjects, api: session.api)
            }
            if detectObjects.isEmpty {
                Text("No objects selected — this camera records only (no detection events).")
                    .font(.caption2)
                    .foregroundStyle(Theme.textSecondary)
            }

            labeledField("Camera AI detection") {
                Picker("Detect mode", selection: $detectMode) {
                    ForEach(DetectMode.allCases, id: \.self) { mode in
                        Text(detectModeLabel(mode)).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
            }
            Text(detectModeCaption)
                .font(.caption2)
                .foregroundStyle(Theme.textSecondary)
            if !camera.capabilities.aiOnCamera {
                Text("This camera has no on-board AI, so the camera-triggered modes fall back to continuous server detection.")
                    .font(.caption2)
                    .foregroundStyle(Theme.warning)
            }

            labeledField("Detection frame rate (fps)") {
                HStack {
                    Slider(value: $detectFps, in: 1 ... 10, step: 1).tint(Theme.accent)
                    Text("\(Int(detectFps))")
                        .font(.caption.weight(.semibold).monospacedDigit())
                        .foregroundStyle(Theme.accent)
                        .frame(width: 24)
                }
            }
            Text("Frames per second analyzed by the detector (1–10; 5 is plenty).")
                .font(.caption2)
                .foregroundStyle(Theme.textSecondary)

            labeledField("Live-view audio") {
                Picker("Live-view audio", selection: $audioCodec) {
                    Text("G.711 (works in live view)").tag("g711a")
                    Text("AAC (higher quality, no live audio)").tag("aac")
                }
                .pickerStyle(.menu)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            Text("G.711 keeps this camera's audio in a WebRTC-legal codec, so live-view sound plays. AAC records at higher quality but live view has no audio.")
                .font(.caption2)
                .foregroundStyle(Theme.textSecondary)

            if camera.capabilities.whiteLight {
                Toggle("Smart spotlight", isOn: $smartSpotlight)
                    .tint(Theme.accent)
                Text("Turn the spotlight on when a person is seen at night; off after the hold below.")
                    .font(.caption2)
                    .foregroundStyle(Theme.textSecondary)
                if smartSpotlight {
                    Stepper("Spotlight hold: \(spotlightHoldSeconds)s",
                            value: $spotlightHoldSeconds, in: 5 ... 600, step: 15)
                    Text("How long the spotlight stays on after the last person is seen (5–600s).")
                        .font(.caption2)
                        .foregroundStyle(Theme.textSecondary)
                }
            }

            saveButton("Save detection", busy: configSaving) {
                Task { await saveCameraConfig() }
            }
        }
    }

    private func detectModeLabel(_ mode: DetectMode) -> String {
        switch mode {
        case .always: return "Server"
        case .cameraAi: return "Camera-triggered"
        case .cameraAiOnly: return "Camera-only"
        }
    }

    private var detectModeCaption: String {
        switch detectMode {
        case .always:
            return "Server always analyzes this camera's video."
        case .cameraAi:
            return "Server analyzes only while the camera's own AI fires (big GPU savings)."
        case .cameraAiOnly:
            return "No server analysis — events come straight from the camera's AI."
        }
    }

    // MARK: - Streams

    private var streamsCard: some View {
        settingsCard("Streams", systemImage: "dot.radiowaves.left.and.right") {
            labeledField("Main stream URL override") {
                TextField("rtsp:// (blank = Amcrest default)", text: $mainUrl)
                    .textFieldStyle(.roundedBorder)
                    .autocorrectionDisabled()
                    .textInputAutocapitalization(.never)
            }
            Text("Recording & live view. Leave empty to use the Amcrest default.")
                .font(.caption2)
                .foregroundStyle(Theme.textSecondary)
            labeledField("Substream URL override") {
                TextField("rtsp:// (blank = Amcrest default)", text: $subUrl)
                    .textFieldStyle(.roundedBorder)
                    .autocorrectionDisabled()
                    .textInputAutocapitalization(.never)
            }
            Text("Detection. Leave empty to use the Amcrest default.")
                .font(.caption2)
                .foregroundStyle(Theme.textSecondary)
            saveButton("Save streams", busy: configSaving) {
                Task { await saveCameraConfig() }
            }
        }
    }

    // MARK: - Exempt zones

    private var zonesCard: some View {
        settingsCard("Detection exempt zones", systemImage: "hand.raised.slash.fill") {
            ExemptZonesEditorView(
                cameraName: camera.name,
                zones: $exemptZones,
                api: session.api
            )
            // Explainer sits UNDER the picture — keeps the frame the first thing
            // you see instead of pushing it down behind a paragraph.
            Text("Draw polygons over the live view. Anything whose feet land inside a zone is ignored for detection — no event, notification or annotation.")
                .font(.caption2)
                .foregroundStyle(Theme.textSecondary)
            saveButton("Save zones", busy: configSaving) {
                Task { await saveCameraConfig() }
            }
        }
    }

    // MARK: - Credentials

    private var credentialsCard: some View {
        settingsCard("Credentials", systemImage: "key.fill") {
            if camera.needsCredentials {
                Label("No camera password stored", systemImage: "exclamationmark.triangle.fill")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(Theme.warning)
            }
            Text("The camera's own username/password — used for night vision, spotlight, siren, talk and model detection. Blank fields keep the stored values.")
                .font(.caption2)
                .foregroundStyle(Theme.textSecondary)
            labeledField("Username") {
                TextField("unchanged if blank", text: $credUser)
                    .textFieldStyle(.roundedBorder)
                    .autocorrectionDisabled()
                    .textInputAutocapitalization(.never)
            }
            labeledField("Password") {
                SecureField("unchanged if blank", text: $credPass)
                    .textFieldStyle(.roundedBorder)
            }
            saveButton("Save & Test", busy: credTesting) {
                Task { await saveAndTestCredentials() }
            }
            if let probe {
                if probe.ok {
                    Label("Connected — model \(probe.model ?? "unknown")",
                          systemImage: "checkmark.seal.fill")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(Theme.success)
                } else {
                    Label(probe.detail ?? "Probe failed", systemImage: "xmark.seal.fill")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(Theme.dangerSoft)
                }
            }
        }
    }

    // MARK: - Device settings

    private var deviceCard: some View {
        settingsCard("Device settings", systemImage: "gearshape.2.fill") {
            if !isOnline {
                Text("Camera is offline — device settings are unavailable.")
                    .font(.caption)
                    .foregroundStyle(Theme.warning)
            } else if deviceLoading && device == nil {
                HStack(spacing: 8) {
                    ProgressView().tint(Theme.accent)
                    Text("Reading device state…")
                        .font(.caption)
                        .foregroundStyle(Theme.textSecondary)
                }
            } else if let deviceError, device == nil {
                Text(deviceError)
                    .font(.caption)
                    .foregroundStyle(Theme.dangerSoft)
            } else if let device {
                // Each control shows only when the device actually reported that
                // field (the GET omits unsupported ones) — parity with web.
                if device.flip != nil {
                    Toggle("Flip image 180°", isOn: $flip)
                        .tint(Theme.accent)
                }

                if device.osdName != nil {
                    labeledField("On-screen display name") {
                        TextField("Camera name overlay", text: $osdName)
                            .textFieldStyle(.roundedBorder)
                            .autocorrectionDisabled()
                    }
                }

                if device.motionDetect != nil {
                    Toggle("On-camera motion detection", isOn: $motionDetect)
                        .tint(Theme.accent)
                }

                if device.volume?.mic != nil {
                    labeledField("Microphone volume: \(Int(micVolume))") {
                        Slider(value: $micVolume, in: 0 ... 100, step: 1).tint(Theme.accent)
                    }
                }
                if device.volume?.speaker != nil {
                    labeledField("Speaker volume: \(Int(speakerVolume))") {
                        Slider(value: $speakerVolume, in: 0 ... 100, step: 1).tint(Theme.accent)
                    }
                }

                saveButton("Apply to device", busy: deviceSaving) {
                    Task { await saveDeviceSettings() }
                }
            }
        }
        .disabled(!isOnline)
        .opacity(isOnline ? 1 : 0.55)
    }

    // MARK: - Seeding + loading

    private func seedFromCamera() {
        friendlyName = camera.friendlyName
        model = camera.model.isEmpty ? "unknown" : camera.model
        ip = camera.ip
        detectObjects = camera.detectObjects
        detectMode = camera.effectiveDetectMode
        audioCodec = camera.effectiveAudioCodec
        smartSpotlight = camera.effectiveSmartSpotlight
        spotlightHoldSeconds = camera.effectiveSpotlightHoldSeconds
        detectFps = Double(camera.detectFps)
        mainUrl = camera.mainUrl
        subUrl = camera.subUrl
        exemptZones = camera.exemptZones ?? []
    }

    private func loadDeviceSettings() async {
        guard session.isAdmin, isOnline, let api = session.api else { return }
        deviceLoading = true
        defer { deviceLoading = false }
        do {
            let settings = try await api.cameraDeviceSettings(camera.name)
            device = settings
            deviceError = nil
            syncDeviceControls(from: settings)
        } catch {
            session.handleAPIError(error)
            deviceError = (error as? ApiError)?.message ?? error.localizedDescription
        }
    }

    private func syncDeviceControls(from settings: DeviceSettings) {
        flip = settings.flip ?? false
        osdName = settings.osdName ?? ""
        motionDetect = settings.motionDetect ?? false
        micVolume = Double(settings.volume?.mic ?? 0)
        speakerVolume = Double(settings.volume?.speaker ?? 0)
    }

    // MARK: - Save actions

    /// One PUT carrying every camera-config concern (identity + detection +
    /// streams + zones), with blank creds so stored device credentials survive.
    private func saveCameraConfig() async {
        guard let api = session.api, !configSaving else { return }
        configSaving = true
        defer { configSaving = false }
        let payload = CameraUpdatePayload(
            name: camera.name,
            friendlyName: friendlyName.trimmingCharacters(in: .whitespaces),
            model: model,
            ip: ip.trimmingCharacters(in: .whitespaces),
            detectObjects: detectObjects,
            exemptZones: exemptZones,
            detectFps: Int(detectFps),
            detectMode: detectMode.rawValue,
            mainUrl: mainUrl.trimmingCharacters(in: .whitespaces),
            subUrl: subUrl.trimmingCharacters(in: .whitespaces),
            audioCodec: audioCodec,
            smartSpotlight: smartSpotlight,
            spotlightHoldSeconds: spotlightHoldSeconds
        )
        do {
            _ = try await api.updateCamera(payload)
            showBanner("Camera settings saved — streams are reloading.")
        } catch {
            session.handleAPIError(error)
            errorMessage = friendly(error, verb: "save the camera settings")
        }
    }

    /// Save the camera's own credentials, then probe (model detection + caps).
    private func saveAndTestCredentials() async {
        guard let api = session.api, !credTesting else { return }
        credTesting = true
        probe = nil
        defer { credTesting = false }
        let payload = CameraUpdatePayload(
            name: camera.name,
            friendlyName: friendlyName.trimmingCharacters(in: .whitespaces),
            model: model,
            ip: ip.trimmingCharacters(in: .whitespaces),
            username: credUser,
            password: credPass
        )
        do {
            _ = try await api.updateCamera(payload)
            let result = try await api.probeCamera(camera.name)
            probe = result
            if result.ok {
                credPass = ""
                await loadDeviceSettings()
            }
        } catch {
            session.handleAPIError(error)
            errorMessage = friendly(error, verb: "save and test the credentials")
        }
    }

    /// Apply the device-level settings via the sparse settings PATCH.
    private func saveDeviceSettings() async {
        guard let api = session.api, let device, !deviceSaving else { return }
        deviceSaving = true
        defer { deviceSaving = false }
        // Send only the fields the device reported (present == supported), so we
        // never push an unsupported knob it would reject.
        var patch = DeviceSettingsPatch()
        if device.flip != nil { patch.flip = flip }
        if device.osdName != nil { patch.osdName = osdName }
        if device.motionDetect != nil { patch.motionDetect = motionDetect }
        if device.volume?.mic != nil { patch.volume = .init(mic: Int(micVolume), speaker: patch.volume?.speaker) }
        if device.volume?.speaker != nil {
            patch.volume = .init(mic: patch.volume?.mic, speaker: Int(speakerVolume))
        }
        do {
            let updated = try await api.updateCameraDeviceSettings(camera.name, patch: patch)
            self.device = updated
            syncDeviceControls(from: updated)
            showBanner("Device settings applied.")
        } catch {
            session.handleAPIError(error)
            errorMessage = friendly(error, verb: "apply the device settings")
        }
    }

    private func showBanner(_ text: String) {
        withAnimation { banner = text }
        Task {
            try? await Task.sleep(nanoseconds: 2_500_000_000)
            if banner == text { withAnimation { banner = nil } }
        }
    }

    private func friendly(_ error: Error, verb: String) -> String {
        if let apiError = error as? ApiError {
            if apiError.status == 403 { return "Admin access is required to \(verb)." }
            if apiError.status == 501 { return "The camera firmware doesn't support this." }
            return "Couldn't \(verb): \(apiError.message)"
        }
        return "Couldn't \(verb): \(error.localizedDescription)"
    }

    // MARK: - Small builders

    private func settingsCard<Content: View>(
        _ title: String, systemImage: String, @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Label(title, systemImage: systemImage)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(Theme.textPrimary)
            content()
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.cardBackground())
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    private func labeledField<Content: View>(
        _ label: String, @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(label)
                .font(.caption.weight(.semibold))
                .foregroundStyle(Theme.textSecondary)
            content()
        }
    }

    private func saveButton(_ title: String, busy: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 8) {
                if busy { ProgressView().tint(.white) }
                Text(busy ? "Saving…" : title)
            }
            .font(.callout.weight(.semibold))
            .foregroundStyle(.white)
            .frame(maxWidth: .infinity)
            .frame(height: 44)
            .background(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(Theme.accent.opacity(busy ? 0.6 : 1))
            )
        }
        .buttonStyle(.plain)
        .disabled(busy)
    }
}

// MARK: - Object picker

/// Per-camera "Detect objects" picker. Fetches the active detector's label
/// vocabulary (falling back to a bundled COCO-80 list), shows the current
/// selection as removable chips, and offers a searchable/free-text add — a
/// pragmatic parity with the web ObjectPicker.
private struct ObjectPickerView: View {
    @Binding var selected: [String]
    let api: APIClient?

    @State private var vocabulary: [String] = ObjectPickerView.cocoLabels
    @State private var query = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Selected chips (tap to remove)
            if selected.isEmpty {
                Text("None selected")
                    .font(.caption2)
                    .foregroundStyle(Theme.textSecondary)
            } else {
                FlowChips(items: selected) { label in
                    ChipView(text: label, selected: true) { remove(label) }
                }
            }

            TextField("Search classes…", text: $query)
                .textFieldStyle(.roundedBorder)
                .autocorrectionDisabled()
                .textInputAutocapitalization(.never)
                .onSubmit(commitFreeText)

            // Candidate matches (tap to add)
            if !matches.isEmpty {
                FlowChips(items: matches) { label in
                    ChipView(text: label, selected: false) { add(label) }
                }
            } else if canAddFree {
                ChipView(text: "+ Add \"\(normalized(query))\"", selected: false) {
                    commitFreeText()
                }
            }
        }
        .task { await loadLabels() }
    }

    private var matches: [String] {
        let q = query.trimmingCharacters(in: .whitespaces)
            .lowercased().replacingOccurrences(of: "_", with: " ")
        let sel = Set(selected)
        let pool = vocabulary.filter { !sel.contains($0) }
        let filtered = q.isEmpty ? pool : pool.filter {
            $0.lowercased().replacingOccurrences(of: "_", with: " ").contains(q)
        }
        return Array(filtered.prefix(30))
    }

    private func normalized(_ s: String) -> String {
        s.trimmingCharacters(in: .whitespaces).lowercased()
            .replacingOccurrences(of: " ", with: "_")
    }

    private var canAddFree: Bool {
        let n = normalized(query)
        return !n.isEmpty && !selected.contains(n)
    }

    private func add(_ label: String) {
        if !label.isEmpty, !selected.contains(label) { selected.append(label) }
    }

    private func remove(_ label: String) {
        selected.removeAll { $0 == label }
    }

    private func commitFreeText() {
        if let first = matches.first {
            add(first)
            query = ""
        } else if canAddFree {
            add(normalized(query))
            query = ""
        }
    }

    private func loadLabels() async {
        guard let api else { return }
        if let resp = try? await api.detectionLabels(), !resp.labels.isEmpty {
            vocabulary = resp.labels
        }
    }

    /// Bundled COCO-80 fallback (used when GET /api/detection/labels is absent).
    static let cocoLabels: [String] = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic_light", "fire_hydrant", "stop_sign",
        "parking_meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
        "tie", "suitcase", "frisbee", "skis", "snowboard", "sports_ball", "kite",
        "baseball_bat", "baseball_glove", "skateboard", "surfboard",
        "tennis_racket", "bottle", "wine_glass", "cup", "fork", "knife", "spoon",
        "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
        "hot_dog", "pizza", "donut", "cake", "chair", "couch", "potted_plant",
        "bed", "dining_table", "toilet", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell_phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy_bear",
        "hair_drier", "toothbrush",
    ]
}

/// A single selectable/removable chip.
private struct ChipView: View {
    let text: String
    let selected: Bool
    let onTap: () -> Void

    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 4) {
                Text(text)
                    .font(.caption.weight(.medium))
                if selected {
                    Image(systemName: "xmark")
                        .font(.system(size: 9, weight: .bold))
                }
            }
            .foregroundStyle(selected ? Theme.accent : Theme.textPrimary)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(
                Capsule().fill(selected ? Theme.accent.opacity(0.15)
                                        : Theme.textSecondary.opacity(0.10))
            )
            .overlay(
                Capsule().stroke(
                    selected ? Theme.accent.opacity(0.5) : Theme.border,
                    lineWidth: 1
                )
            )
        }
        .buttonStyle(.plain)
    }
}

/// Minimal wrapping chip layout (SwiftUI `Layout`, iOS 16+).
private struct FlowChips<Item: Hashable, Cell: View>: View {
    let items: [Item]
    @ViewBuilder let cell: (Item) -> Cell

    var body: some View {
        FlowLayout(spacing: 6) {
            ForEach(items, id: \.self) { item in
                cell(item)
            }
        }
    }
}

/// A simple flow (wrapping) layout used by the chip clouds.
private struct FlowLayout: Layout {
    var spacing: CGFloat = 6

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout Void) -> CGSize {
        let maxWidth = proposal.width ?? .infinity
        var rowWidth: CGFloat = 0
        var rowHeight: CGFloat = 0
        var totalHeight: CGFloat = 0
        var totalWidth: CGFloat = 0
        for sub in subviews {
            let size = sub.sizeThatFits(.unspecified)
            if rowWidth + size.width > maxWidth, rowWidth > 0 {
                totalHeight += rowHeight + spacing
                totalWidth = max(totalWidth, rowWidth - spacing)
                rowWidth = 0
                rowHeight = 0
            }
            rowWidth += size.width + spacing
            rowHeight = max(rowHeight, size.height)
        }
        totalHeight += rowHeight
        totalWidth = max(totalWidth, rowWidth - spacing)
        return CGSize(width: min(totalWidth, maxWidth), height: totalHeight)
    }

    func placeSubviews(
        in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout Void
    ) {
        var x = bounds.minX
        var y = bounds.minY
        var rowHeight: CGFloat = 0
        for sub in subviews {
            let size = sub.sizeThatFits(.unspecified)
            if x + size.width > bounds.maxX, x > bounds.minX {
                x = bounds.minX
                y += rowHeight + spacing
                rowHeight = 0
            }
            sub.place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(size))
            x += size.width + spacing
            rowHeight = max(rowHeight, size.height)
        }
    }
}

// MARK: - Exempt zones editor

/// Draw / manage per-camera exempt detection zones over a live snapshot. Points
/// are kept NORMALIZED (0…1) against the displayed frame so they survive any
/// resolution change (parity with the web ExemptZonesEditor). Tap to add draft
/// points; Finish commits a polygon (≥3 points); existing zones can be removed.
private struct ExemptZonesEditorView: View {
    let cameraName: String
    @Binding var zones: [ExemptZone]
    let api: APIClient?

    @State private var draft: [[Double]] = []

    private let zoneColors: [Color] = [.red, .orange, .blue, .purple, .pink, .green]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            GeometryReader { geo in
                ZStack {
                    Rectangle().fill(Theme.bgDeep)
                    if let url = api?.cameraSnapshotURL(cameraName) {
                        AsyncImage(url: url) { phase in
                            switch phase {
                            case .success(let image):
                                image.resizable().scaledToFill()
                            case .failure:
                                snapshotUnavailable
                            default:
                                ProgressView().tint(Theme.accent)
                            }
                        }
                    } else {
                        snapshotUnavailable
                    }

                    zoneOverlay(size: geo.size)
                }
                .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                .contentShape(Rectangle())
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onEnded { value in
                            addPoint(value.location, in: geo.size)
                        }
                )
            }
            .frame(height: 180)

            HStack(spacing: 8) {
                smallButton("Finish zone (\(draft.count))", disabled: draft.count < 3) {
                    finishZone()
                }
                smallButton("Undo point", disabled: draft.isEmpty) {
                    if !draft.isEmpty { draft.removeLast() }
                }
                smallButton("Clear draft", disabled: draft.isEmpty) {
                    draft.removeAll()
                }
            }

            if zones.isEmpty {
                Text("No exempt zones — the whole frame is watched.")
                    .font(.caption2)
                    .foregroundStyle(Theme.textSecondary)
            } else {
                ForEach(Array(zones.enumerated()), id: \.offset) { idx, zone in
                    HStack(spacing: 8) {
                        Circle()
                            .fill(zoneColors[idx % zoneColors.count])
                            .frame(width: 12, height: 12)
                        // Editable name — committed by the same "Save zones" PUT
                        // (ExemptZone is Codable, so `name` rides the payload).
                        TextField("Zone \(idx + 1)", text: $zones[idx].name)
                            .font(.caption)
                            .textInputAutocapitalization(.words)
                            .autocorrectionDisabled()
                            .foregroundStyle(Theme.textPrimary)
                        Text("\(zone.points.count) pts")
                            .font(.caption2)
                            .foregroundStyle(Theme.textSecondary)
                        Button("Remove") { zones.remove(at: idx) }
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(Theme.dangerSoft)
                            .buttonStyle(.plain)
                    }
                }
            }
        }
    }

    private var snapshotUnavailable: some View {
        Text("Live view unavailable")
            .font(.caption)
            .foregroundStyle(Theme.textSecondary)
    }

    private func zoneOverlay(size: CGSize) -> some View {
        ZStack {
            ForEach(Array(zones.enumerated()), id: \.offset) { idx, zone in
                polygonPath(zone.points, in: size)
                    .fill(zoneColors[idx % zoneColors.count].opacity(0.28))
                polygonPath(zone.points, in: size)
                    .stroke(zoneColors[idx % zoneColors.count], lineWidth: 2)
            }
            if draft.count >= 2 {
                draftPath(in: size)
                    .stroke(Theme.accent, style: StrokeStyle(lineWidth: 2, dash: [5, 4]))
            }
            ForEach(Array(draft.enumerated()), id: \.offset) { _, pt in
                Circle()
                    .fill(Theme.accent)
                    .frame(width: 8, height: 8)
                    .position(x: pt[0] * size.width, y: pt[1] * size.height)
            }
        }
    }

    private func polygonPath(_ points: [[Double]], in size: CGSize) -> Path {
        Path { p in
            guard let first = points.first else { return }
            p.move(to: CGPoint(x: first[0] * size.width, y: first[1] * size.height))
            for pt in points.dropFirst() {
                p.addLine(to: CGPoint(x: pt[0] * size.width, y: pt[1] * size.height))
            }
            p.closeSubpath()
        }
    }

    private func draftPath(in size: CGSize) -> Path {
        Path { p in
            guard let first = draft.first else { return }
            p.move(to: CGPoint(x: first[0] * size.width, y: first[1] * size.height))
            for pt in draft.dropFirst() {
                p.addLine(to: CGPoint(x: pt[0] * size.width, y: pt[1] * size.height))
            }
        }
    }

    private func addPoint(_ location: CGPoint, in size: CGSize) {
        guard size.width > 0, size.height > 0 else { return }
        let x = min(1, max(0, Double(location.x / size.width)))
        let y = min(1, max(0, Double(location.y / size.height)))
        draft.append([(x * 10000).rounded() / 10000, (y * 10000).rounded() / 10000])
    }

    private func finishZone() {
        guard draft.count >= 3 else { return }
        zones.append(ExemptZone(name: "Zone \(zones.count + 1)", points: draft))
        draft.removeAll()
    }

    private func smallButton(_ title: String, disabled: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(disabled ? Theme.textSecondary : Theme.accent)
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(Theme.accent.opacity(disabled ? 0.05 : 0.12))
                )
        }
        .buttonStyle(.plain)
        .disabled(disabled)
    }
}

// MARK: - Small compatibility helper

/// Lightweight stand-in for ContentUnavailableView so the admin-guard message
/// renders identically across the deployment target.
private struct ContentUnavailableCompat: View {
    let title: String
    let systemImage: String
    let message: String

    var body: some View {
        VStack(spacing: 10) {
            Image(systemName: systemImage)
                .font(.system(size: 40, weight: .semibold))
                .foregroundStyle(Theme.textSecondary)
            Text(title)
                .font(.headline)
                .foregroundStyle(Theme.textPrimary)
            Text(message)
                .font(.caption)
                .foregroundStyle(Theme.textSecondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(32)
    }
}
