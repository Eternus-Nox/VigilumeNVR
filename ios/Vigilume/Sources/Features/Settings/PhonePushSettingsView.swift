import SwiftUI

/// Settings › Phone push: the iOS twin of the web's Settings → Notifications
/// "Phone push" card (frontend/src/pages/settings/NotificationsTab.tsx).
///
/// Two independent channels behind a segmented picker:
/// - **Vigilume iOS app** (`notifications.apns`) — delivery to THIS app via the
///   push relay: a native notification and the CallKit doorbell ring.
/// - **ntfy** (`notifications.ntfy`) — push with no Apple developer account and
///   no relay, at the cost of the ring (alerts land in the ntfy app).
///
/// **THE PICKER IS A VIEW SWITCH, NOT A MODE SWITCH.** Both channels can be on
/// at once and fire off the same rules; the picker only chooses which one you
/// are editing. Each panel carries its own toggle and the picker labels show
/// both channels' live state, so the hidden one can never be silently off. Do
/// not collapse this into a single three-way mode — that would make enabling
/// ntfy disable the doorbell ring.
///
/// **Every write here is PATCH /api/settings — NEVER PUT.** PUT is a
/// full-document replace and every field carries a backend default, so any key
/// the body omits is RESET rather than left alone (this is not hypothetical: a
/// PUT missing `notifications.apns.direct.p8` destroyed the APNs signing key
/// back when that field existed). Save sends ONLY the subtree being edited.
///
/// Admin-only: reached from `SettingsHomeView`'s `session.isAdmin` section, so
/// it lives inside that view's existing NavigationStack — no NavigationStack of
/// its own.
struct PhonePushSettingsView: View {
    @EnvironmentObject private var session: SessionModel

    private enum Channel: String, CaseIterable {
        case apns, ntfy
    }

    /// Last document loaded from the server — the baseline the drafts diff
    /// against to decide whether Save is enabled.
    @State private var doc: SettingsDocument?
    @State private var loading = true
    @State private var loadError: String?
    @State private var channel: Channel = .apns

    // Draft — APNs
    @State private var apnsOn = false
    @State private var relayUrl = ""

    // Draft — ntfy
    @State private var enabled = false
    @State private var server = "https://ntfy.sh"
    @State private var topic = ""
    @State private var authToken = ""
    @State private var priority = 4
    @State private var attachSnapshot = true

    // Event-type filters that apply to whichever channel(s) are on.
    @State private var cameraDownAlerts = false

    // Draft — rules (the `notifications` subtree's channel-independent knobs:
    // they gate EVERY channel, so they're seeded in apply() and saved by the
    // same Save button as apns/ntfy — never patched on their own). Note
    // `notificationsEnabled` is the master switch, distinct from `enabled`
    // above, which is the ntfy channel's own toggle.
    @State private var notificationsEnabled = true
    @State private var drawBoxes = true
    @State private var labels: [String] = []
    @State private var cooldownSeconds = 60
    @State private var minScore = 0.7

    /// Count of registered APNs devices (GET .../apns/devices) — a read-only
    /// reassurance under the iOS-app section. nil = not loaded or an older
    /// backend without the route, so the readout stays hidden rather than
    /// showing a misleading 0.
    @State private var apnsDeviceCount: Int?

    @State private var saving = false
    @State private var saveError: String?
    @State private var copied = false

    /// What to put in Relay URL when the NVR runs the bundled `push-relay`
    /// itself — the Docker-internal service name, NOT the public hostname a
    /// tunnel points at. Going out to the internet and back means push breaks
    /// whenever the tunnel, DNS, or the connection does.
    private static let localRelayUrl = "http://push-relay:8090"

    /// The backend validates `priority` with `Field(ge=1, le=5)` — ntfy's own
    /// scale. Bound to the Picker so the UI can't produce a 422.
    private static let priorities: [(Int, String)] = [
        (1, "1 — Min"), (2, "2 — Low"), (3, "3 — Default"), (4, "4 — High"), (5, "5 — Urgent"),
    ]

    /// Backend bounds for the rules the UI sends, so the controls can't produce
    /// a 422: cooldown 0…86400s (the Stepper range clamps), minScore 0…1.
    private static let cooldownRange = 0 ... 86400
    private static let minScoreRange = 0.0 ... 1.0

    var body: some View {
        Group {
            if let loadError, doc == nil {
                ContentUnavailableView(
                    "Couldn't load settings",
                    systemImage: "externaldrive.badge.xmark",
                    description: Text(loadError)
                )
            } else {
                form
            }
        }
        .background(Theme.bg)
        .navigationTitle("Phone push")
        .navigationBarTitleDisplayMode(.inline)
        .task { await load() }
        .refreshable { await load() }
    }

    private var form: some View {
        List {
            rulesSection
            eventAlertsSection
            channelSection
            if channel == .apns {
                apnsEnableSection
                if apnsOn { relaySection }
            } else {
                enableSection
                serverSection
                topicSection
                authSection
                deliverySection
            }
            saveSection
        }
        .scrollContentBackground(.hidden)
        .overlay {
            if loading && doc == nil {
                ProgressView().tint(Theme.accent)
            }
        }
        .disabled(doc == nil)
    }

    // MARK: - Rules (channel-independent; the notifications subtree)

    /// The rules a detection must clear before ANY channel notifies — the master
    /// switch, which objects qualify, a per-camera cooldown, a confidence floor,
    /// and whether snapshots get detection boxes drawn on. All live in the
    /// `notifications` subtree, so they ride the same Save button as the channels
    /// below (never patched on their own). Sits at the top: it governs what the
    /// channels below deliver.
    private var rulesSection: some View {
        Section {
            Toggle("Send notifications", isOn: $notificationsEnabled)
                .tint(Theme.accent)

            Toggle("Draw detection boxes on snapshots", isOn: $drawBoxes)
                .tint(Theme.accent)

            VStack(alignment: .leading, spacing: 6) {
                Text("Which objects notify")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(Theme.textSecondary)
                NotifyLabelsField(labels: $labels)
            }
            .padding(.vertical, 2)

            cooldownRow
            minScoreRow
        } header: {
            Text("When to notify")
        } footer: {
            Text("These rules apply to every channel below — a detection has to match one of these objects and clear the score and cooldown before any phone is alerted. The master switch turns all notifications off without changing anything else.")
        }
        .listRowBackground(Theme.surface)
    }

    /// Cooldown between alerts, mirroring RecordingSettingsView's retention
    /// Steppers: a 0…86400 Stepper (the range clamps, so no 422) with a
    /// human-readable value and a hint. Stepped 5s at a time — 1s granularity
    /// across a 24-hour range would be unusable.
    private var cooldownRow: some View {
        VStack(alignment: .leading, spacing: 4) {
            Stepper(value: $cooldownSeconds, in: Self.cooldownRange, step: 5) {
                HStack {
                    Text("Cooldown between alerts")
                        .foregroundStyle(Theme.textPrimary)
                    Spacer()
                    Text(cooldownText(cooldownSeconds))
                        .font(.subheadline.weight(.medium).monospacedDigit())
                        .foregroundStyle(cooldownSeconds == 0 ? Theme.warning : Theme.textPrimary)
                }
            }
            .tint(Theme.accent)
            .accessibilityValue(cooldownText(cooldownSeconds))

            Text(cooldownSeconds == 0
                 ? "No cooldown — every qualifying detection can notify."
                 : "Minimum gap before the same camera notifies again.")
                .font(.caption)
                .foregroundStyle(cooldownSeconds == 0 ? Theme.warning : Theme.textSecondary)
        }
        .padding(.vertical, 2)
    }

    /// Confidence floor for a detection to notify, mirroring SystemSettingsView's
    /// Confidence slider: 0…1 in 0.05 steps, shown as a percentage.
    private var minScoreRow: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Minimum score")
                    .foregroundStyle(Theme.textSecondary)
                Spacer()
                Text("\(Int((minScore * 100).rounded()))%")
                    .font(.subheadline.weight(.medium).monospacedDigit())
                    .foregroundStyle(Theme.textPrimary)
            }
            Slider(value: $minScore, in: Self.minScoreRange, step: 0.05)
                .tint(Theme.accent)
                .accessibilityLabel("Minimum notification score")
                .accessibilityValue("\(Int((minScore * 100).rounded())) percent")
        }
        .padding(.vertical, 2)
    }

    /// Seconds as a short human string: "Off", "45s", "1m", "1m 30s", "2h 5m".
    private func cooldownText(_ seconds: Int) -> String {
        if seconds == 0 { return "Off" }
        let h = seconds / 3600
        let m = (seconds % 3600) / 60
        let s = seconds % 60
        var parts: [String] = []
        if h > 0 { parts.append("\(h)h") }
        if m > 0 { parts.append("\(m)m") }
        if s > 0 { parts.append("\(s)s") }
        return parts.joined(separator: " ")
    }

    // MARK: - Event-type alerts (channel-independent)

    /// Which system events (beyond detections) generate a push. Applies to
    /// whichever channel is on, so it lives above the channel picker. Saved by
    /// the same Save button as the rest of the form.
    private var eventAlertsSection: some View {
        Section {
            Toggle("A camera goes offline", isOn: $cameraDownAlerts)
                .tint(Theme.accent)
        } header: {
            Text("Also alert me when")
        } footer: {
            Text("Sends a push if a camera stops responding for about a minute (debounced, so a brief blip stays quiet), and again when it comes back. Independent of detection alerts.")
        }
        .listRowBackground(Theme.surface)
    }

    // MARK: - Channel picker

    private var channelSection: some View {
        Section {
            Picker("Channel", selection: $channel) {
                Text(apnsOn ? "iOS app · On" : "iOS app · Off").tag(Channel.apns)
                Text(enabled ? "ntfy · On" : "ntfy · Off").tag(Channel.ntfy)
            }
            .pickerStyle(.segmented)
            .listRowInsets(EdgeInsets(top: 8, leading: 12, bottom: 8, trailing: 12))
        } footer: {
            Text("Two independent ways to reach a phone, both firing off the same rules and per-camera filters. Switching tabs only changes which one you're editing — you can run both at once.")
        }
        .listRowBackground(Theme.surface)
    }

    // MARK: - APNs sections

    private var apnsEnableSection: some View {
        Section {
            Toggle("Send notifications to the Vigilume iOS app", isOn: $apnsOn)
                .tint(Theme.accent)
            // Read-only reassurance: how many phones are registered to receive
            // the push. Hidden unless the backend answered the devices route.
            if let apnsDeviceCount {
                HStack {
                    Text("Registered devices")
                        .foregroundStyle(Theme.textPrimary)
                    Spacer()
                    Text("\(apnsDeviceCount) registered")
                        .font(.subheadline.monospacedDigit())
                        .foregroundStyle(Theme.textSecondary)
                }
            }
        } footer: {
            Text("The real thing: a native notification and a doorbell that rings like a phone call on a locked screen. Delivery goes through a relay holding the app's Apple signing key — this NVR never needs an Apple developer account. Payloads are encrypted end-to-end, so the relay can't read them, and snapshots never pass through it.")
        }
        .listRowBackground(Theme.surface)
    }

    private var relaySection: some View {
        Section {
            TextField(Self.localRelayUrl, text: $relayUrl)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .keyboardType(.URL)
                .font(.system(.body, design: .monospaced))
                .foregroundStyle(Theme.textPrimary)
            Button("Use local relay") {
                relayUrl = Self.localRelayUrl
            }
            .foregroundStyle(Theme.accent)
        } header: {
            Text("Relay URL")
        } footer: {
            if relayUrl.trimmed.isEmpty {
                Text("No relay URL set — nothing will be delivered to the iOS app until you fill this in.")
                    .foregroundStyle(Theme.danger)
            } else {
                Text("Running the bundled relay on the NVR? Use \(Self.localRelayUrl) — the container name, not your public hostname. Going out to the internet and back means push breaks whenever your tunnel, DNS, or connection does. Otherwise, the address of the relay you were given.")
            }
        }
        .listRowBackground(Theme.surface)
    }

    // MARK: - ntfy sections

    private var enableSection: some View {
        Section {
            Toggle("Send notifications to ntfy", isOn: $enabled)
                .tint(Theme.accent)
        } footer: {
            Text("Push with no Apple developer account and no relay — install the ntfy app, subscribe to the topic below, done. The trade: alerts arrive in the ntfy app, so there's no doorbell ring and no Vigilume UI around them.")
        }
        .listRowBackground(Theme.surface)
    }

    private var serverSection: some View {
        Section {
            TextField("https://ntfy.sh", text: $server)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .keyboardType(.URL)
                .foregroundStyle(Theme.textPrimary)
        } header: {
            Text("Server")
        } footer: {
            Text("Public ntfy.sh, or your own server. Self-hosting with auth-default-access: deny-all plus an access token below is the private option.")
        }
        .listRowBackground(Theme.surface)
    }

    private var topicSection: some View {
        Section {
            TextField("tap Generate", text: $topic)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .font(.system(.body, design: .monospaced))
                .foregroundStyle(Theme.textPrimary)
            HStack {
                Button("Generate") {
                    topic = Self.generateTopic()
                    copied = false
                }
                .foregroundStyle(Theme.accent)
                Spacer()
                Button(copied ? "Copied" : "Copy") {
                    UIPasteboard.general.string = topic
                    copied = true
                    Task {
                        try? await Task.sleep(for: .seconds(2))
                        copied = false
                    }
                }
                .foregroundStyle(topic.isEmpty ? Theme.textSecondary : Theme.accent)
                .disabled(topic.isEmpty)
            }
        } header: {
            Text("Topic")
        } footer: {
            // Not a nag: on a default-allow server this string is the ONLY
            // thing gating access to every notification, and an attached
            // snapshot URL carries a media token.
            Text("Treat this like a password. On ntfy.sh (and any server with default access) anyone who knows the topic receives every notification from this NVR — including the snapshot links. Use Generate rather than a name someone could guess, and only share it with your own devices.")
        }
        .listRowBackground(Theme.surface)
    }

    private var authSection: some View {
        Section {
            SecureField("tk_… (optional)", text: $authToken)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .foregroundStyle(Theme.textPrimary)
        } header: {
            Text("Access token")
        } footer: {
            Text("Required by a self-hosted server with deny-all access, and for reserved topics on ntfy.sh. Leave empty for an open server.")
        }
        .listRowBackground(Theme.surface)
    }

    private var deliverySection: some View {
        Section {
            Picker("Priority", selection: $priority) {
                ForEach(Self.priorities, id: \.0) { value, label in
                    Text(label).tag(value)
                }
            }
            Toggle("Attach the event snapshot", isOn: $attachSnapshot)
                .tint(Theme.accent)
        } header: {
            Text("Delivery")
        } footer: {
            Text("The image is linked, never uploaded — your phone fetches it straight from this NVR, so it never touches the ntfy server. Needs Settings → System → Public URL to be reachable from your phone. Turn this off for text-only notifications.")
        }
        .listRowBackground(Theme.surface)
    }

    private var saveSection: some View {
        Section {
            Button {
                Task { await save() }
            } label: {
                HStack {
                    Spacer()
                    if saving {
                        ProgressView().tint(Theme.accent)
                    } else {
                        Text("Save phone push settings")
                            .foregroundStyle(dirty ? Theme.accent : Theme.textSecondary)
                    }
                    Spacer()
                }
            }
            .disabled(saving || !dirty)

            if let saveError {
                Text(saveError)
                    .font(.footnote)
                    .foregroundStyle(Theme.danger)
            }
        } footer: {
            Text("Saves both channels — nothing else on the server is touched.")
        }
        .listRowBackground(Theme.surface)
    }

    // MARK: - Topic

    /// A fresh, unguessable topic.
    ///
    /// A SECURITY control, not a convenience: on a default-allow ntfy server
    /// the topic is the only thing between a stranger and every notification
    /// this NVR sends. A human-chosen topic ("vigilume", "home") is guessable
    /// in seconds, so the UI never offers an empty box to type in.
    ///
    /// 128 bits from `SystemRandomNumberGenerator` (arc4random on Apple
    /// platforms — cryptographically secure), hex, inside the backend's
    /// `^[A-Za-z0-9_-]{1,64}$`. Mirrors the web's generateTopic().
    static func generateTopic() -> String {
        let hex = (0..<16)
            .map { _ in String(format: "%02x", UInt8.random(in: UInt8.min...UInt8.max)) }
            .joined()
        return "vigilume_\(hex)"
    }

    // MARK: - State

    private var apnsDirty: Bool {
        guard let a = doc?.notifications.apns else { return false }
        return (a.mode == .relay) != apnsOn || a.relayUrl != relayUrl.trimmed
    }

    private var ntfyDirty: Bool {
        guard let n = doc?.notifications.ntfy else { return false }
        return n.enabled != enabled
            || n.server != server.trimmed
            || n.topic != topic.trimmed
            || n.authToken != authToken.trimmed
            || n.priority != priority
            || n.attachSnapshot != attachSnapshot
    }

    private var cameraDownDirty: Bool {
        guard let doc else { return false }
        return doc.notifications.cameraDownAlerts != cameraDownAlerts
    }

    // Rules — one comparison per field so Save sends only what actually changed
    // (mirrors cameraDownDirty), and so `rulesDirty` can OR them for the button.

    private var notificationsEnabledDirty: Bool {
        guard let doc else { return false }
        return doc.notifications.enabled != notificationsEnabled
    }

    private var labelsDirty: Bool {
        guard let doc else { return false }
        return doc.notifications.labels != labels
    }

    private var cooldownDirty: Bool {
        guard let doc else { return false }
        return doc.notifications.cooldownSeconds != cooldownSeconds
    }

    private var minScoreDirty: Bool {
        guard let doc else { return false }
        return abs(doc.notifications.minScore - minScore) > 0.0001
    }

    private var drawBoxesDirty: Bool {
        guard let doc else { return false }
        return doc.notifications.drawBoxes != drawBoxes
    }

    private var rulesDirty: Bool {
        notificationsEnabledDirty || labelsDirty || cooldownDirty
            || minScoreDirty || drawBoxesDirty
    }

    private var dirty: Bool { apnsDirty || ntfyDirty || cameraDownDirty || rulesDirty }

    private func save() async {
        guard let api = session.api, !saving else { return }
        saving = true
        defer { saving = false }
        saveError = nil
        // Only the subtrees that actually changed. The backend deep-merges, and
        // every field here is Optional (nil is omitted by JSONEncoder), so an
        // untouched channel is left exactly as stored rather than rewritten.
        let patch = SettingsPatch(
            notifications: .init(
                apns: apnsDirty ? .init(
                    mode: apnsOn ? "relay" : "off",
                    // Trim + strip a trailing slash so {relay_url}/api/push
                    // can't double up; the backend does the same, this just
                    // keeps the field showing what was actually stored.
                    relayUrl: relayUrl.trimmed.replacingOccurrences(
                        of: "/+$", with: "", options: .regularExpression
                    )
                ) : nil,
                ntfy: ntfyDirty ? .init(
                    enabled: enabled,
                    server: server.trimmed.replacingOccurrences(
                        of: "/+$", with: "", options: .regularExpression
                    ),
                    topic: topic.trimmed,
                    authToken: authToken.trimmed,
                    priority: priority,
                    attachSnapshot: attachSnapshot
                ) : nil,
                cameraDownAlerts: cameraDownDirty ? cameraDownAlerts : nil,
                // Rules ride this same PATCH — each nil-omitted unless changed,
                // so an untouched rule is left exactly as stored (like the
                // channels above). `labels` is already lowercased/capped/deduped
                // by the editor, so it's sent as-is.
                enabled: notificationsEnabledDirty ? notificationsEnabled : nil,
                labels: labelsDirty ? labels : nil,
                cooldownSeconds: cooldownDirty ? cooldownSeconds : nil,
                minScore: minScoreDirty ? minScore : nil,
                drawBoxes: drawBoxesDirty ? drawBoxes : nil
            )
        )
        do {
            apply(try await api.patchSettings(patch))
        } catch {
            session.handleAPIError(error)
            // A bad relay URL / server / topic is a readable 422 from the
            // backend's validators.
            saveError = (error as? ApiError)?.message ?? error.localizedDescription
        }
    }

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
        await loadApnsDevices()
    }

    /// Registered-device count for the readout under the iOS-app section. A
    /// 404-tolerant read: an older backend without the route just leaves the
    /// readout hidden (nil), and a transient failure keeps the last good count.
    private func loadApnsDevices() async {
        guard let api = session.api else { return }
        if let devices = try? await api.apnsDevices() {
            apnsDeviceCount = devices.count
        }
    }

    /// Adopt a server document (initial load or a PATCH response) as both the
    /// baseline and the draft — the response is the full updated document, so a
    /// save leaves the form showing exactly what the server now holds.
    private func apply(_ document: SettingsDocument) {
        let first = doc == nil
        doc = document
        let a = document.notifications.apns
        apnsOn = a.mode == .relay
        relayUrl = a.relayUrl
        let n = document.notifications.ntfy
        enabled = n.enabled
        server = n.server
        topic = n.topic
        authToken = n.authToken
        priority = n.priority
        attachSnapshot = n.attachSnapshot
        cameraDownAlerts = document.notifications.cameraDownAlerts
        notificationsEnabled = document.notifications.enabled
        drawBoxes = document.notifications.drawBoxes
        labels = document.notifications.labels
        cooldownSeconds = document.notifications.cooldownSeconds
        minScore = document.notifications.minScore
        // Open on a channel that's already configured, so an admin coming back
        // to check a setting lands on it; otherwise APNs, the one that rings.
        // Only on first load — never yank the picker out from under an edit.
        if first {
            channel = (a.mode == .relay || !n.enabled) ? .apns : .ntfy
        }
    }
}

private extension String {
    var trimmed: String { trimmingCharacters(in: .whitespacesAndNewlines) }
}

// MARK: - Labels editor

/// The "which objects notify" chip editor — removable chips plus a free-text
/// add, with one-tap suggestions. The visual is deliberately identical to the
/// per-camera "Detect objects" picker (CameraSettingsView's ObjectPickerView):
/// that one's chip components are private to their own file, so the look is
/// mirrored here rather than reused. Entries are lowercased, capped at 32 chars
/// and deduped on add — the backend's own rule for a label.
private struct NotifyLabelsField: View {
    @Binding var labels: [String]
    @State private var draft = ""

    /// The handful the web seeds a fresh install with.
    private static let suggestions = ["person", "dog", "cat", "car"]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Selected chips (tap to remove).
            if labels.isEmpty {
                Text("No objects — nothing will notify until you add one.")
                    .font(.caption2)
                    .foregroundStyle(Theme.danger)
            } else {
                FlowChips(items: labels) { label in
                    ChipView(text: label, selected: true) { remove(label) }
                }
            }

            TextField("Add an object…", text: $draft)
                .textFieldStyle(.roundedBorder)
                .autocorrectionDisabled()
                .textInputAutocapitalization(.never)
                .onSubmit(commit)

            // Suggestions not already chosen (tap to add).
            let unused = Self.suggestions.filter { !labels.contains($0) }
            if !unused.isEmpty {
                FlowChips(items: unused) { label in
                    ChipView(text: label, selected: false) { add(label) }
                }
            }
        }
    }

    /// Lowercase + cap at 32 chars — matches the backend's per-label rule.
    private func normalized(_ s: String) -> String {
        String(s.trimmingCharacters(in: .whitespaces).lowercased().prefix(32))
    }

    private func add(_ raw: String) {
        let value = normalized(raw)
        if !value.isEmpty, !labels.contains(value) { labels.append(value) }
    }

    private func remove(_ label: String) {
        labels.removeAll { $0 == label }
    }

    private func commit() {
        add(draft)
        draft = ""
    }
}

// MARK: - Chip components (mirrored from CameraSettingsView's private set)

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
