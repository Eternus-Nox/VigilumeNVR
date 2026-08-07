import SwiftUI
import UserNotifications

/// The public repository bug reports are filed against. ONE literal, in ONE
/// place: a fork changes this ONE literal and nowhere else, so never inline
/// this URL at a call site.
private let vigilumeRepoURL = "https://github.com/Eternus-Nox/VigilumeNVR"

/// Percent-encoding set for a single query VALUE.
///
/// `.urlQueryAllowed` permits `&`, `+`, `=` and `?` — legal in a query *string*,
/// fatal inside one value: the first `&` in the issue body would truncate
/// everything after it and every `+` would arrive as a space. Subtract them so
/// the whole template survives the round trip.
private let issueQueryAllowed: CharacterSet = {
    var set = CharacterSet.urlQueryAllowed
    set.remove(charactersIn: "&+=?")
    return set
}()

/// The Settings tab content: account, push notifications, servers
/// (add/switch/remove), About (app + server health), and — for admins — a
/// link-out note that full system configuration lives in the web app.
struct SettingsHomeView: View {
    @EnvironmentObject private var session: SessionModel
    @EnvironmentObject private var serverStore: ServerStore
    @ObservedObject var push: PushManager
    /// Drives the live "On LAN / Remote" video-path badge on the active server.
    @ObservedObject private var lan = LANReachability.shared
    @Environment(\.openURL) private var openURL

    @State private var showingAddServer = false
    /// Drives the self-service "Change password" sheet (all roles).
    @State private var showingChangePassword = false
    /// Server whose video routing is being edited.
    @State private var routingServer: ServerConfig?
    @State private var health: SystemHealth?
    @State private var healthError: String?
    @State private var isSigningOut = false
    @State private var confirmingPurge: PurgeKind?
    @State private var isPurging = false
    @State private var purgeResult: String?

    /// Which irreversible admin bulk-delete is being confirmed.
    private enum PurgeKind: Identifiable {
        case events, recordings
        var id: Int { self == .events ? 0 : 1 }
        var confirmLabel: String {
            switch self {
            case .events: return "Delete All Events"
            case .recordings: return "Delete All Recordings"
            }
        }
        var message: String {
            switch self {
            case .events:
                return "Permanently delete every event, including all snapshots and clips. This cannot be undone."
            case .recordings:
                return "Permanently delete all continuous recorded footage for every camera. Recording resumes immediately. This cannot be undone."
            }
        }
    }

    var body: some View {
        NavigationStack {
            List {
                accountSection
                notificationsSection
                serversSection
                // Groups are require_auth, not admin — the web shows them to
                // viewers too (it's a viewer's default settings tab), so this
                // sits OUTSIDE the isAdmin gate below.
                groupsSection
                aboutSection
                if session.isAdmin, let webURL = serverStore.activeServer?.url {
                    adminSection(webURL: webURL)
                }
                if session.isAdmin {
                    detectionSection
                }
                if session.isAdmin {
                    alertsSection
                }
                if session.isAdmin {
                    usersSection
                }
                if session.isAdmin {
                    dangerZoneSection
                }
                signOutSection
            }
            .scrollContentBackground(.hidden)
            .background(Theme.bg)
            .navigationTitle("Settings")
            .sheet(isPresented: $showingAddServer) {
                AddServerSheet()
            }
            .sheet(isPresented: $showingChangePassword) {
                ChangePasswordSheet()
            }
            .sheet(item: $routingServer) { server in
                VideoRoutingSheet(server: server)
            }
            .task {
                await push.refreshAuthorization()
                await loadHealth()
            }
            .refreshable {
                await loadHealth()
            }
            .confirmationDialog(
                confirmingPurge?.confirmLabel ?? "",
                isPresented: Binding(
                    get: { confirmingPurge != nil },
                    set: { if !$0 { confirmingPurge = nil } }
                ),
                titleVisibility: .visible,
                presenting: confirmingPurge
            ) { kind in
                Button(kind.confirmLabel, role: .destructive) {
                    Task { await runPurge(kind) }
                }
                Button("Cancel", role: .cancel) {}
            } message: { kind in
                Text(kind.message)
            }
            .alert(
                "Done",
                isPresented: Binding(
                    get: { purgeResult != nil },
                    set: { if !$0 { purgeResult = nil } }
                )
            ) {
                Button("OK", role: .cancel) {}
            } message: {
                Text(purgeResult ?? "")
            }
        }
    }

    // MARK: Account

    private var accountSection: some View {
        Section("Account") {
            row("User", session.username ?? "—")
            row("Role", session.role?.rawValue.capitalized ?? "—")
            row("Server", serverStore.activeServer?.name ?? "—")
            liveUpdatesRow
            // Self-service, so it's outside every admin gate — any signed-in
            // user changes their own password. The built-in env admin gets a
            // 400, surfaced readably in the sheet rather than hidden here.
            Button {
                showingChangePassword = true
            } label: {
                Label("Change password", systemImage: "key.fill")
                    .foregroundStyle(Theme.accent)
            }
        }
        .listRowBackground(Theme.surface)
    }

    private var liveUpdatesRow: some View {
        HStack {
            Text("Live updates")
                .foregroundStyle(Theme.textSecondary)
            Spacer()
            switch session.socket.state {
            case .connected:
                Label("Connected", systemImage: "circle.fill")
                    .font(.footnote)
                    .foregroundStyle(Theme.success)
            case .connecting:
                Label("Connecting", systemImage: "circle.fill")
                    .font(.footnote)
                    .foregroundStyle(Theme.warning)
            case .disconnected:
                Label("Offline", systemImage: "circle.fill")
                    .font(.footnote)
                    .foregroundStyle(Theme.danger)
            }
        }
    }

    // MARK: Notifications

    private var pushToggleBinding: Binding<Bool> {
        Binding(
            get: { push.isEnabled },
            set: { on in
                Task {
                    if on {
                        await push.enable()
                    } else {
                        await push.disable()
                    }
                }
            }
        )
    }

    private var pushBusy: Bool {
        push.status == .requestingPermission || push.status == .registering
    }

    private var notificationsSection: some View {
        Section {
            Toggle(isOn: pushToggleBinding) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Push notifications")
                        .foregroundStyle(Theme.textPrimary)
                    Text("Event alerts with snapshots on this device")
                        .font(.caption)
                        .foregroundStyle(Theme.textSecondary)
                }
            }
            .disabled(pushBusy)

            HStack {
                Text("Status")
                    .foregroundStyle(Theme.textSecondary)
                Spacer()
                Text(pushStatusText)
                    .font(.footnote)
                    .foregroundStyle(pushStatusColor)
                    .multilineTextAlignment(.trailing)
            }

            if push.status == .permissionDenied {
                Button {
                    if let url = URL(string: UIApplication.openSettingsURLString) {
                        openURL(url)
                    }
                } label: {
                    Label("Open iOS Settings", systemImage: "gear")
                        .foregroundStyle(Theme.accent)
                }
            }
        } header: {
            Text("Notifications")
        } footer: {
            if push.status == .serverUnsupported {
                Text("This Vigilume server hasn't been updated for native push yet. This device will register automatically once the server adds APNs support.")
            }
        }
        .listRowBackground(Theme.surface)
    }

    private var pushStatusText: String {
        switch push.status {
        case .off: return "Off"
        case .requestingPermission: return "Requesting permission…"
        case .awaitingToken: return "Waiting for Apple push token…"
        case .registering: return "Registering with server…"
        case .registered: return "Active on this device"
        case .permissionDenied: return "Denied in iOS Settings"
        case .serverUnsupported: return "Server not updated yet"
        case .failed(let message): return message
        }
    }

    private var pushStatusColor: Color {
        switch push.status {
        case .registered:
            return Theme.success
        case .permissionDenied, .failed:
            return Theme.danger
        case .serverUnsupported, .awaitingToken:
            return Theme.warning
        case .off, .requestingPermission, .registering:
            return Theme.textSecondary
        }
    }

    // MARK: Servers

    private var serversSection: some View {
        Section {
            ForEach(serverStore.servers) { server in
                serverRow(server)
            }
            .onDelete(perform: deleteServers)

            Button {
                showingAddServer = true
            } label: {
                Label("Add Server", systemImage: "plus.circle.fill")
                    .foregroundStyle(Theme.accent)
            }
        } header: {
            Text("Servers")
        } footer: {
            if serverStore.servers.count > 1 {
                Text("Tap a server to switch. Swipe to remove a saved server — the active one can't be removed while signed in.")
            }
        }
        .listRowBackground(Theme.surface)
    }

    private func serverRow(_ server: ServerConfig) -> some View {
        Button {
            guard server.id != serverStore.activeServerID else { return }
            Task {
                await session.switchServer(to: server.id)
                push.syncToActiveServer()
                await loadHealth()
            }
        } label: {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(server.name)
                        .foregroundStyle(Theme.textPrimary)
                    HStack(spacing: 4) {
                        if server.isInsecureHTTP {
                            Image(systemName: "lock.open.fill")
                                .font(.caption2)
                                .foregroundStyle(Theme.warning)
                        }
                        Text(server.urlString)
                            .font(.caption)
                            .foregroundStyle(Theme.textSecondary)
                    }
                    videoPathBadge(server)
                }
                Spacer()
                if server.id == serverStore.activeServerID {
                    Image(systemName: "checkmark")
                        .foregroundStyle(Theme.accent)
                }
            }
        }
        .deleteDisabled(server.id == serverStore.activeServerID)
        // Video routing is per-server config, not a row action, so it lives in
        // a swipe/long-press affordance rather than stealing the row's tap
        // (which switches servers).
        .swipeActions(edge: .leading, allowsFullSwipe: false) {
            Button {
                routingServer = server
            } label: {
                Label("Video routing", systemImage: "wifi.router")
            }
            .tint(Theme.accent)
        }
        .contextMenu {
            Button {
                routingServer = server
            } label: {
                Label("Video routing…", systemImage: "wifi.router")
            }
        }
    }

    /// Live indicator of which network path video is taking for the ACTIVE
    /// server that has a LAN address configured: on the LAN (fast, direct) vs
    /// remote over the primary URL. Reachability is only tracked for the active
    /// server, so nothing shows for the others.
    @ViewBuilder
    private func videoPathBadge(_ server: ServerConfig) -> some View {
        if server.id == serverStore.activeServerID {
            switch lan.mediaRoute(for: server.id) {
            case .lan(let url):
                Label("Video direct — \(url.host ?? "local")",
                      systemImage: "bolt.horizontal.fill")
                    .font(.caption2)
                    .foregroundStyle(Theme.success)
            case .none:
                Label("Video via \(server.url?.host ?? "main address")",
                      systemImage: "network")
                    .font(.caption2)
                    .foregroundStyle(Theme.textSecondary)
            }
        }
    }

    private func deleteServers(at offsets: IndexSet) {
        for offset in offsets {
            let server = serverStore.servers[offset]
            guard server.id != serverStore.activeServerID else { continue }
            serverStore.removeServer(id: server.id)
        }
    }

    // MARK: Camera groups (all roles)

    /// `/api/groups` is `require_auth`, so viewers get this too — groups are a
    /// property of the server, shared by everyone who signs in.
    private var groupsSection: some View {
        Section {
            NavigationLink {
                GroupsView()
            } label: {
                Label("Camera groups", systemImage: "rectangle.3.group.fill")
                    .foregroundStyle(Theme.textPrimary)
            }
        } footer: {
            Text("Group cameras into the filter chips on the Cameras tab. Groups are shared with everyone on this server.")
        }
        .listRowBackground(Theme.surface)
    }

    // MARK: About

    private var aboutSection: some View {
        Section("About") {
            row("App version", appVersion)
            if let health {
                row("Server version", health.version)
                HStack {
                    Text("Detector")
                        .foregroundStyle(Theme.textSecondary)
                    Spacer()
                    Text(detectorText(health.detector))
                        .foregroundStyle(health.detector.ready ? Theme.success : Theme.warning)
                }
                HStack {
                    Text("go2rtc")
                        .foregroundStyle(Theme.textSecondary)
                    Spacer()
                    Text(health.go2rtc ? "OK" : "Down")
                        .foregroundStyle(health.go2rtc ? Theme.success : Theme.danger)
                }
                row("Cameras online", String(health.camerasOnline))
            } else if let healthError {
                HStack {
                    Text("Server health")
                        .foregroundStyle(Theme.textSecondary)
                    Spacer()
                    Text(healthError)
                        .font(.footnote)
                        .foregroundStyle(Theme.danger)
                        .multilineTextAlignment(.trailing)
                }
            } else {
                HStack {
                    Text("Server health")
                        .foregroundStyle(Theme.textSecondary)
                    Spacer()
                    ProgressView()
                        .tint(Theme.accent)
                }
            }
            NavigationLink {
                CameraHealthView()
            } label: {
                Label("Camera health", systemImage: "waveform.path.ecg")
                    .foregroundStyle(Theme.textPrimary)
            }
            reportBugRow
            legalLinks
        }
        .listRowBackground(Theme.surface)
    }

    // MARK: Report a bug

    /// Opens a PRE-FILLED issue on the public tracker.
    ///
    /// WHY NO BACKEND. Vigilume is self-hosted — a report posted to *your own*
    /// NVR reaches nobody, so there is deliberately no endpoint behind this: it
    /// is a `github.com/…/issues/new` URL built from what this screen already
    /// knows, and the reporter's own GitHub session does the posting. Nothing
    /// leaves the device until they press Submit, and they can edit every
    /// diagnostic first.
    ///
    /// Sits in About, outside every `isAdmin` gate — a viewer hits bugs too.
    @ViewBuilder
    private var reportBugRow: some View {
        if let url = bugReportURL {
            Button {
                openURL(url)
            } label: {
                HStack {
                    Label("Report a bug", systemImage: "ladybug")
                        .foregroundStyle(Theme.accent)
                    Spacer()
                    Image(systemName: "arrow.up.right.square")
                        .imageScale(.medium)
                        .foregroundStyle(Theme.textSecondary)
                }
            }
        }
    }

    private var bugReportURL: URL? {
        guard
            // "[iOS]" so the tracker can tell an app report from a web one at a
            // glance; the trailing space is where the reporter starts typing.
            let title = "[iOS] ".addingPercentEncoding(withAllowedCharacters: issueQueryAllowed),
            let body = bugReportBody.addingPercentEncoding(withAllowedCharacters: issueQueryAllowed)
        else { return nil }
        return URL(string: "\(vigilumeRepoURL)/issues/new?title=\(title)&body=\(body)")
    }

    /// The pre-filled issue body. Diagnostics we cannot actually read are
    /// OMITTED rather than printed as "nil", which would read like a real
    /// answer to whoever triages it. `health` is nil until `loadHealth()`
    /// returns (or when the server is unreachable — often the very bug being
    /// reported), so the server lines degrade to "unknown" / drop out.
    private var bugReportBody: String {
        var lines = [
            "### What happened?",
            "<!-- describe the bug -->",
            "",
            "### Steps to reproduce",
            "1.",
            "2.",
            "",
            "### Environment (auto-filled, please keep)",
            // `appVersion` is already "1.2 (34)" — version and build in one.
            "- App: iOS \(appVersion)",
            "- Device: \(deviceModel) / iOS \(UIDevice.current.systemVersion)",
            "- Server version: \(health?.version ?? "unknown")",
        ]
        if let detector = health?.detector {
            lines.append("- Detector: \(detectorText(detector))")
        }
        return lines.joined(separator: "\n")
    }

    /// Hardware identifier, e.g. `iPhone15,2`. `UIDevice.current.model` only
    /// ever says "iPhone", which tells a triager nothing; this names the actual
    /// hardware. Falls back to the generic name if `uname` returns nothing.
    private var deviceModel: String {
        var info = utsname()
        uname(&info)
        let machine = withUnsafeBytes(of: info.machine) { raw in
            String(decoding: raw.prefix { $0 != 0 }, as: UTF8.self)
        }
        return machine.isEmpty ? UIDevice.current.model : machine
    }

    /// Privacy Policy / Terms of Use — served by the SAME server the app is
    /// connected to (origin derived from the API base URL), so they always
    /// point at the active NVR's /privacy.html and /terms.html.
    @ViewBuilder
    private var legalLinks: some View {
        if let origin = legalOrigin {
            legalLinkRow("Privacy Policy", url: origin.appendingPathComponent("privacy.html"))
            legalLinkRow("Terms of Use", url: origin.appendingPathComponent("terms.html"))
        }
    }

    /// Scheme + host (+ port) of the configured server, stripping any path/query
    /// so we can append the legal page filenames cleanly.
    private var legalOrigin: URL? {
        guard let base = session.api?.apiBase,
              var comps = URLComponents(url: base, resolvingAgainstBaseURL: false)
        else { return nil }
        comps.path = ""
        comps.query = nil
        comps.fragment = nil
        return comps.url
    }

    private func legalLinkRow(_ title: String, url: URL) -> some View {
        Link(destination: url) {
            HStack {
                Text(title)
                    .foregroundStyle(Theme.accent)
                Spacer()
                Image(systemName: "arrow.up.right.square")
                    .imageScale(.medium)
                    .foregroundStyle(Theme.textSecondary)
            }
        }
    }

    private var appVersion: String {
        let version = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "?"
        let build = Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "?"
        return "\(version) (\(build))"
    }

    private func detectorText(_ detector: SystemHealth.Detector) -> String {
        guard detector.ready else { return "Not ready" }
        var parts = ["Ready"]
        // Friendly device names, matching the web's describeDetector — the raw
        // backend values are "cuda" / "cpu" / "edgetpu".
        if let device = detector.device {
            switch device {
            case "edgetpu": parts.append("Coral Edge TPU")
            case "cuda": parts.append("GPU")
            default: parts.append(device.uppercased())
            }
        }
        if let model = detector.model { parts.append(model) }
        return parts.joined(separator: " · ")
    }

    private func loadHealth() async {
        guard let api = session.api else { return }
        do {
            health = try await api.systemHealth()
            healthError = nil
        } catch {
            health = nil
            healthError = (error as? ApiError)?.message ?? error.localizedDescription
        }
    }

    // MARK: Admin link-out

    private func adminSection(webURL: URL) -> some View {
        Section("Administration") {
            Link(destination: webURL) {
                HStack(alignment: .center) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Open web app")
                            .foregroundStyle(Theme.accent)
                        Text("Model downloads and the rest of the server configuration live in the Vigilume web app.")
                            .font(.caption)
                            .foregroundStyle(Theme.textSecondary)
                    }
                    Spacer()
                    // Center the glyph in a square frame so it sits on the row's
                    // vertical axis rather than hugging the leading text's cap.
                    Image(systemName: "arrow.up.right.square")
                        .imageScale(.large)
                        .frame(width: 24, height: 24, alignment: .center)
                        .foregroundStyle(Theme.textSecondary)
                }
            }
            NavigationLink {
                IntegrationsView()
            } label: {
                Label("Integrations", systemImage: "house.fill")
                    .foregroundStyle(Theme.textPrimary)
            }
        }
        .listRowBackground(Theme.surface)
    }

    // MARK: Detection (admin-only)

    private var detectionSection: some View {
        Section("Cameras & Detection") {
            NavigationLink {
                CamerasAdminView()
            } label: {
                Label("Cameras", systemImage: "video.fill")
                    .foregroundStyle(Theme.textPrimary)
            }
            NavigationLink {
                PrivacyModeView()
            } label: {
                Label("Privacy Mode", systemImage: "eye.slash.fill")
                    .foregroundStyle(Theme.textPrimary)
            }
            NavigationLink {
                SuppressionsView()
            } label: {
                Label("Excluded objects", systemImage: "eye.slash")
                    .foregroundStyle(Theme.textPrimary)
            }
            NavigationLink {
                RecordingSettingsView()
            } label: {
                Label("Recording", systemImage: "externaldrive.fill")
                    .foregroundStyle(Theme.textPrimary)
            }
            NavigationLink {
                SystemSettingsView()
            } label: {
                Label("System", systemImage: "gearshape.2.fill")
                    .foregroundStyle(Theme.textPrimary)
            }
            NavigationLink {
                DetectorStatusView()
            } label: {
                Label("Detector status", systemImage: "cpu.fill")
                    .foregroundStyle(Theme.textPrimary)
            }
        }
        .listRowBackground(Theme.surface)
    }

    // MARK: Alerts (admin-only)

    /// Server-wide notification CHANNELS, as opposed to the per-device push
    /// toggle in `notificationsSection` above — that one is viewer-visible
    /// (any role registers its own device); these write the settings document,
    /// which is `require_admin`.
    ///
    /// Both phone-push channels live behind one screen: APNs (this app, via the
    /// relay — the only route to the CallKit doorbell ring) and ntfy (any
    /// phone, no Apple account, no ring).
    ///
    /// APNs used to be web-only, on the grounds that configuring it meant
    /// pasting Apple's .p8 and a private key has no business on a phone. That
    /// reason died with `direct` mode: the block is now a toggle and a relay
    /// URL, with no secret in it.
    private var alertsSection: some View {
        Section("Alerts") {
            NavigationLink {
                PhonePushSettingsView()
            } label: {
                Label("Phone push", systemImage: "bell.badge.fill")
                    .foregroundStyle(Theme.textPrimary)
            }
        }
        .listRowBackground(Theme.surface)
    }

    // MARK: Users (admin-only)

    /// `/api/users` is `require_admin`, so this row only exists for admins — and
    /// `UsersView` re-checks `session.isAdmin` for itself.
    private var usersSection: some View {
        Section {
            NavigationLink {
                UsersView()
            } label: {
                Label("Users", systemImage: "person.2.fill")
                    .foregroundStyle(Theme.textPrimary)
            }
        } header: {
            Text("Access")
        } footer: {
            Text("Create admin or viewer accounts for this server. The built-in admin is set by the server's ADMIN_PASSWORD and isn't managed here.")
        }
        .listRowBackground(Theme.surface)
    }

    // MARK: Danger zone (admin-only, irreversible)

    private var dangerZoneSection: some View {
        Section {
            Button(role: .destructive) {
                confirmingPurge = .events
            } label: {
                dangerRow(title: "Delete all events",
                          subtitle: "Removes every event, snapshot and clip.")
            }
            .disabled(isPurging)
            Button(role: .destructive) {
                confirmingPurge = .recordings
            } label: {
                dangerRow(title: "Delete all recordings",
                          subtitle: "Removes all 24/7 recorded footage.")
            }
            .disabled(isPurging)
        } header: {
            Text("Danger Zone")
        } footer: {
            Text("These permanently delete data for every camera and cannot be undone.")
        }
        .listRowBackground(Theme.surface)
    }

    private func dangerRow(title: String, subtitle: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(title)
                if isPurging {
                    Spacer()
                    ProgressView()
                }
            }
            Text(subtitle)
                .font(.caption)
                .foregroundStyle(Theme.textSecondary)
        }
    }

    private func runPurge(_ kind: PurgeKind) async {
        guard session.isAdmin, let api = session.api else { return }
        isPurging = true
        defer { isPurging = false }
        do {
            switch kind {
            case .events:
                try await api.deleteAllEvents()
                purgeResult = "All events have been deleted."
            case .recordings:
                try await api.deleteAllRecordings()
                purgeResult = "All recordings have been deleted."
            }
        } catch {
            purgeResult = "Delete failed: \(error.localizedDescription)"
        }
    }

    // MARK: Sign out

    private var signOutSection: some View {
        Section {
            Button(role: .destructive) {
                guard !isSigningOut else { return }
                isSigningOut = true
                Task {
                    // Unregister push while the session token is still valid.
                    await push.unregisterForLogout()
                    session.logout()
                    isSigningOut = false
                }
            } label: {
                HStack {
                    Spacer()
                    if isSigningOut {
                        ProgressView()
                    } else {
                        Text("Sign Out")
                    }
                    Spacer()
                }
            }
        }
        .listRowBackground(Theme.surface)
    }

    // MARK: Shared row

    private func row(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label)
                .foregroundStyle(Theme.textSecondary)
            Spacer()
            Text(value)
                .foregroundStyle(Theme.textPrimary)
        }
    }
}

// MARK: - Change password sheet

/// Self-service password change for the signed-in account, any role:
/// POST /api/users/me/password → 204. The bounds mirror the backend validator
/// (new is 8…256 chars) so the common mistakes are caught before the round-trip,
/// but the server's `detail` is what's shown when one slips through — notably the
/// built-in env admin's 400 ("built-in admin can't change password here"), which
/// we surface plainly rather than pretending it worked.
private struct ChangePasswordSheet: View {
    @EnvironmentObject private var session: SessionModel
    @Environment(\.dismiss) private var dismiss

    static let minLength = 8
    static let maxLength = 256

    @State private var current = ""
    @State private var newPassword = ""
    @State private var confirm = ""
    @State private var saving = false
    @State private var errorMessage: String?
    /// Drives the success confirmation; its OK button dismisses the sheet.
    @State private var didChange = false

    private var newLengthOK: Bool {
        newPassword.count >= Self.minLength && newPassword.count <= Self.maxLength
    }
    private var matches: Bool { newPassword == confirm }

    private var canSave: Bool {
        !current.isEmpty && newLengthOK && matches && !saving
    }

    /// Live, specific hint — only once there's something to complain about.
    private var newPasswordProblem: String? {
        if !newPassword.isEmpty && !newLengthOK {
            return "\(Self.minLength)–\(Self.maxLength) characters."
        }
        if !confirm.isEmpty && !matches {
            return "The new passwords don't match."
        }
        return nil
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    SecureField("Current password", text: $current)
                        .textContentType(.password)
                } header: {
                    Text("Current")
                }
                .listRowBackground(Theme.surface)

                Section {
                    SecureField("At least \(Self.minLength) characters", text: $newPassword)
                        .textContentType(.newPassword)
                    SecureField("Confirm new password", text: $confirm)
                        .textContentType(.newPassword)
                } header: {
                    Text("New password")
                } footer: {
                    if let newPasswordProblem {
                        Text(newPasswordProblem).foregroundStyle(Theme.danger)
                    } else {
                        Text("At least \(Self.minLength) characters. You'll stay signed in on this device after changing it.")
                    }
                }
                .listRowBackground(Theme.surface)

                if let errorMessage {
                    Section {
                        Text(errorMessage)
                            .font(.footnote)
                            .foregroundStyle(Theme.danger)
                    }
                    .listRowBackground(Theme.surface)
                }
            }
            .scrollContentBackground(.hidden)
            .background(Theme.bg)
            .navigationTitle("Change Password")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                        .disabled(saving)
                }
                ToolbarItem(placement: .confirmationAction) {
                    if saving {
                        ProgressView().tint(Theme.accent)
                    } else {
                        Button("Save") { Task { await save() } }
                            .disabled(!canSave)
                    }
                }
            }
            .alert("Password changed", isPresented: $didChange) {
                Button("OK") { dismiss() }
            } message: {
                Text("Your password has been updated. You're still signed in on this device.")
            }
        }
    }

    private func save() async {
        guard canSave else { return }
        saving = true
        defer { saving = false }
        do {
            try await session.api?.changeOwnPassword(current: current, new: newPassword)
            // Clear the fields so nothing lingers behind the confirmation.
            current = ""
            newPassword = ""
            confirm = ""
            errorMessage = nil
            didChange = true
        } catch {
            session.handleAPIError(error)
            // 401 wrong current password, 400 for the built-in env admin
            // ("built-in admin can't change password here"), or the validator's
            // `detail` — surface whichever it is verbatim.
            errorMessage = (error as? ApiError)?.message ?? error.localizedDescription
        }
    }
}
