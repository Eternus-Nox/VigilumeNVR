import SwiftUI
import UIKit   // UIPasteboard, for copying the rclone authorize command

/// Settings › Integrations: the iOS twin of the web's Integrations tab —
/// Home Assistant over MQTT, the cloud storage accounts the nightly archive
/// uploads to, and the archive's own schedule and status.
///
/// The storage section is deliberately a full replacement for `rclone config`:
/// connecting Dropbox or S3 should not require an SSH session on the NVR. The
/// provider catalogue and its form fields are SERVER-DRIVEN (GET
/// /api/integrations/rclone/providers) rather than duplicated here — seven
/// providers' field lists are exactly what drifts when the same list lives in a
/// Swift file and a React file, and a drifted key makes a remote that saves
/// cleanly and never works.
///
/// **Every write here is PATCH /api/settings — NEVER PUT.** PUT is a
/// full-document replace and every field carries a backend default, so any key
/// the body omits is RESET rather than left alone. Save sends ONLY the `mqtt`
/// subtree, and the backend deep-merges it over the stored document.
///
/// **The broker password is a MASKED secret.** GET /api/settings returns
/// `mqtt.password` masked, so the SecureField deliberately loads EMPTY and an
/// empty field means "keep the stored password" — exactly how
/// PhonePushSettingsView treats `ntfy.authToken`. See `save()` / `apply()`.
///
/// Admin-only: reached from `SettingsHomeView`'s admin section, so it lives
/// inside that view's existing NavigationStack — no NavigationStack of its own.
struct IntegrationsView: View {
    @EnvironmentObject private var session: SessionModel

    /// Last document loaded from the server — the baseline the drafts diff
    /// against to decide whether Save is enabled.
    @State private var doc: SettingsDocument?
    @State private var loading = true
    @State private var loadError: String?

    // Draft — MQTT
    @State private var enabled = false
    @State private var host = ""
    /// Held as text so the numeric field can be edited freely; `clampedPort`
    /// turns it into the 1…65535 int the dirty check and save send.
    @State private var portText = String(Self.defaultPort)
    @State private var username = ""
    /// Loads EMPTY on purpose — the server masks the stored secret, so an empty
    /// field is the signal to carry the stored password forward untouched.
    @State private var password = ""
    @State private var discoveryPrefix = "homeassistant"
    @State private var baseTopic = "vigilume"

    // Draft — cloud archive (a SEPARATE settings subtree from mqtt; save()
    // patches only the ones that actually changed).
    @State private var archiveEnabled = false
    @State private var archiveRemote = ""
    @State private var archiveHour = 3
    @State private var archiveKeepDays = 30
    @State private var archiveIncludeSnapshots = true
    @State private var archiveBwlimit = ""
    /// Read-only server state, fetched separately from the settings document.
    @State private var archiveStatus: ArchiveStatus?
    @State private var archiveRunning = false
    @State private var archiveError: String?

    // Cloud storage accounts — the in-app replacement for `rclone config`.
    @State private var providers: [RcloneProvider] = []
    @State private var remotes: [RcloneRemote] = []
    @State private var rcloneAvailable = true
    @State private var newType = ""
    @State private var newName = ""
    @State private var newValues: [String: String] = [:]
    @State private var creating = false
    @State private var busyRemote: String?
    @State private var setupMessage: String?
    @State private var setupOK = false
    /// OAuth providers offer two routes. `.browser` finishes the sign-in on the
    /// NVR itself and needs the operator's own app credentials; `.token` is the
    /// paste-a-blob path, which needs no app registration but does need rclone
    /// on a desktop.
    @State private var authMode: AuthMode = .browser
    @State private var redirectUri = ""
    @State private var signingIn = false

    enum AuthMode: String, CaseIterable { case browser, token }

    // Test-connection state (a draft probe, never a save).
    @State private var testing = false
    @State private var testResult: MqttTestResult?

    @State private var saving = false
    @State private var saveError: String?

    /// Home Assistant's Mosquitto add-on default; also the field default when
    /// the port box is cleared.
    private static let defaultPort = 1883
    /// MQTT port range — the setter/clamp keep senders inside it so the UI can't
    /// produce a 422.
    private static let portRange = 1 ... 65535

    var body: some View {
        Group {
            if !session.isAdmin {
                ContentUnavailableView(
                    "Admins only",
                    systemImage: "lock.fill",
                    description: Text("Integrations are configured by an administrator.")
                )
            } else if let loadError, doc == nil {
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
        .navigationTitle("Integrations")
        .navigationBarTitleDisplayMode(.inline)
        .task {
            await load()
            await loadArchiveStatus()
            await loadRcloneCatalogue()
        }
        .refreshable {
            await load()
            await loadArchiveStatus()
            await loadRcloneCatalogue()
        }
    }

    private var form: some View {
        List {
            enableSection
            brokerSection
            authSection
            topicsSection
            testSection
            storageAccountsSection
            addStorageSection
            archiveSection
            archiveStatusSection
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

    // MARK: - Enable

    private var enableSection: some View {
        Section {
            Toggle("Home Assistant (MQTT)", isOn: $enabled)
                .tint(Theme.accent)
        } footer: {
            Text("Publishes camera and detection state to an MQTT broker for Home Assistant's auto-discovery. Fill in the broker below, test the connection, then save.")
        }
        .listRowBackground(Theme.surface)
    }

    // MARK: - Broker (host + port)

    private var brokerSection: some View {
        Section {
            TextField("192.168.1.10 or broker hostname", text: $host)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .keyboardType(.URL)
                .font(.system(.body, design: .monospaced))
                .foregroundStyle(Theme.textPrimary)

            HStack {
                Text("Port")
                    .foregroundStyle(Theme.textPrimary)
                Spacer()
                TextField(String(Self.defaultPort), text: portBinding)
                    .keyboardType(.numberPad)
                    .multilineTextAlignment(.trailing)
                    .font(.system(.body, design: .monospaced))
                    .foregroundStyle(Theme.textPrimary)
                    .frame(maxWidth: 90)
            }
        } header: {
            Text("Broker")
        } footer: {
            Text("The MQTT broker's address and port. Home Assistant's Mosquitto add-on listens on \(Self.defaultPort) by default.")
        }
        .listRowBackground(Theme.surface)
    }

    // MARK: - Authentication (username + masked password)

    private var authSection: some View {
        Section {
            TextField("username (optional)", text: $username)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .foregroundStyle(Theme.textPrimary)

            SecureField(passwordPlaceholder, text: $password)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .foregroundStyle(Theme.textPrimary)
        } header: {
            Text("Authentication")
        } footer: {
            Text(passwordFooter)
        }
        .listRowBackground(Theme.surface)
    }

    // MARK: - Topics (discovery prefix + base topic)

    private var topicsSection: some View {
        Section {
            labeledField(
                title: "Discovery prefix",
                placeholder: "homeassistant",
                text: $discoveryPrefix
            )
            labeledField(
                title: "Base topic",
                placeholder: "vigilume",
                text: $baseTopic
            )
        } header: {
            Text("Topics")
        } footer: {
            Text("The discovery prefix must match Home Assistant's MQTT discovery prefix (default homeassistant). The base topic namespaces every topic this NVR publishes.")
        }
        .listRowBackground(Theme.surface)
    }

    private func labeledField(
        title: String, placeholder: String, text: Binding<String>
    ) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(Theme.textSecondary)
            TextField(placeholder, text: text)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .font(.system(.body, design: .monospaced))
                .foregroundStyle(Theme.textPrimary)
        }
        .padding(.vertical, 2)
    }

    // MARK: - Test connection

    private var testSection: some View {
        Section {
            Button {
                Task { await test() }
            } label: {
                HStack(spacing: 8) {
                    if testing {
                        ProgressView().tint(Theme.accent)
                    }
                    Text("Test connection")
                        .foregroundStyle(canTest ? Theme.accent : Theme.textSecondary)
                    Spacer()
                }
            }
            .disabled(!canTest)

            if let testResult {
                HStack(spacing: 8) {
                    Image(systemName: testResult.ok
                          ? "checkmark.circle.fill" : "xmark.octagon.fill")
                        .foregroundStyle(testResult.ok ? Theme.success : Theme.danger)
                    Text(testResult.ok
                         ? "Connected"
                         : (testResult.detail?.trimmed.isEmpty == false
                            ? testResult.detail! : "Connection failed"))
                        .font(.footnote)
                        .foregroundStyle(testResult.ok ? Theme.success : Theme.danger)
                }
            }
        } footer: {
            Text("Probes the broker with the values above without saving anything.")
        }
        .listRowBackground(Theme.surface)
    }

    // MARK: - Save

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
                        Text("Save integration settings")
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
            Text("Saves the MQTT integration — nothing else on the server is touched.")
        }
        .listRowBackground(Theme.surface)
    }

    // MARK: - Port helpers

    /// A digits-only, max-5-char text binding for the port field, so a paste
    /// can't sneak in letters or an oversized value.
    private var portBinding: Binding<String> {
        Binding(
            get: { portText },
            set: { portText = String($0.filter(\.isNumber).prefix(5)) }
        )
    }

    /// The port field as the clamped int actually sent/compared. Empty or a
    /// stray 0 falls back to the default; anything above the range is capped.
    private var clampedPort: Int {
        let raw = Int(portText) ?? Self.defaultPort
        return min(max(raw, Self.portRange.lowerBound), Self.portRange.upperBound)
    }

    // MARK: - Password helpers (masked-secret carry-forward)

    /// The server returns a non-empty (masked) string when a password is stored.
    private var hasStoredPassword: Bool { !(doc?.mqtt.password.isEmpty ?? true) }

    private var passwordPlaceholder: String {
        hasStoredPassword ? "Leave blank to keep saved password" : "password (optional)"
    }

    private var passwordFooter: String {
        hasStoredPassword
            ? "A password is already saved. Leave this blank to keep it, or type a new one to replace it."
            : "Optional — only needed if your broker requires authentication."
    }

    /// The user typed a replacement secret. Empty means "keep what's stored".
    private var passwordChanged: Bool { !password.isEmpty }

    // MARK: - Dirty machinery

    private var dirty: Bool { mqttDirty || archiveDirty }

    private var archiveDirty: Bool {
        guard let a = doc?.archive else { return false }
        return a.enabled != archiveEnabled
            || a.remote != archiveRemote.trimmed
            || a.hour != archiveHour
            || a.keepDays != archiveKeepDays
            || a.includeSnapshots != archiveIncludeSnapshots
            || a.bwlimit != archiveBwlimit.trimmed
    }

    private var mqttDirty: Bool {
        guard let m = doc?.mqtt else { return false }
        return m.enabled != enabled
            || m.host != host.trimmed
            || m.port != clampedPort
            || m.username != username.trimmed
            || m.discoveryPrefix != discoveryPrefix.trimmed
            || m.baseTopic != baseTopic.trimmed
            || passwordChanged
    }

    private var canTest: Bool { !host.trimmed.isEmpty && !testing }



    // MARK: - Cloud storage accounts

    private var selectedProvider: RcloneProvider? {
        providers.first { $0.type == newType }
    }

    /// Required fields the operator must actually supply. A required field with
    /// a DEFAULT is already satisfied, so demanding it here would leave the
    /// button dead for every S3 and WebDAV account.
    private var canCreateRemote: Bool {
        guard let p = selectedProvider, !newName.trimmed.isEmpty, !creating else { return false }
        return p.fields
            .filter { $0.required && $0.default.isEmpty }
            .allSatisfy { !(newValues[$0.key] ?? "").trimmed.isEmpty }
    }

    private var storageAccountsSection: some View {
        Section {
            if !rcloneAvailable {
                Text("Cloud storage support is not in this backend build. Rebuild the server and pull to refresh.")
                    .font(.footnote)
                    .foregroundStyle(Theme.warning)
            } else if remotes.isEmpty {
                Text("No storage connected yet.")
                    .font(.footnote)
                    .foregroundStyle(Theme.textSecondary)
            }
            ForEach(remotes) { remote in
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text(remote.name)
                            .foregroundStyle(Theme.textPrimary)
                        Spacer()
                        Text(remote.label)
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(Theme.textSecondary)
                    }
                    HStack(spacing: 14) {
                        Button(busyRemote == remote.name ? "Testing…" : "Test") {
                            Task { await testRemote(remote.name) }
                        }
                        .font(.footnote.weight(.semibold))
                        .foregroundStyle(Theme.accent)
                        .disabled(busyRemote != nil)

                        Button("Forget") {
                            Task { await removeRemote(remote.name) }
                        }
                        .font(.footnote.weight(.semibold))
                        .foregroundStyle(Theme.danger)
                        .disabled(busyRemote != nil)
                    }
                }
                .padding(.vertical, 2)
            }
        } header: {
            Text("Cloud storage accounts")
        } footer: {
            Text("Where the nightly archive uploads. Forgetting an account only removes Vigilume's access — nothing already in the cloud is deleted.")
        }
        .listRowBackground(Theme.surface)
    }

    private var addStorageSection: some View {
        Section {
            Picker("Add storage", selection: $newType) {
                Text("Choose…").tag("")
                ForEach(providers) { p in
                    Text(p.label).tag(p.type)
                }
            }
            .tint(Theme.accent)
            .onChange(of: newType) { _, type in
                // Field keys are per-provider, so carrying values across a
                // switch would submit another backend's keys and be refused.
                newValues = [:]
                setupMessage = nil
                authMode = .browser
                if newName.trimmed.isEmpty { newName = type }
                Task { await loadRedirectUri() }
            }

            if let p = selectedProvider {
                if !p.blurb.isEmpty {
                    Text(p.blurb)
                        .font(.caption)
                        .foregroundStyle(Theme.textSecondary)
                }

                VStack(alignment: .leading, spacing: 4) {
                    Text("Name it").foregroundStyle(Theme.textPrimary)
                    TextField(p.type, text: $newName)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .foregroundStyle(Theme.textPrimary)
                }
                .padding(.vertical, 2)

                if p.oauth {
                    Picker("Sign-in method", selection: $authMode) {
                        Text("Sign in here").tag(AuthMode.browser)
                        Text("Paste a token").tag(AuthMode.token)
                    }
                    .pickerStyle(.segmented)
                    .onChange(of: authMode) { _, _ in
                        setupMessage = nil
                        Task { await loadRedirectUri() }
                    }

                    if authMode == .browser {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Sign-in finishes on the NVR itself, so nothing is needed on a computer. It does need its own free app on \(p.label)'s developer site — create one, add this exact redirect address to it, then paste its key and secret below.")
                                .font(.caption)
                                .foregroundStyle(Theme.textSecondary)
                            if !p.consoleUrl.isEmpty, let url = URL(string: p.consoleUrl) {
                                Link("Open the \(p.label) developer site", destination: url)
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(Theme.accent)
                            }
                            if !redirectUri.isEmpty {
                                HStack {
                                    Text(redirectUri)
                                        .font(.caption.monospaced())
                                        .foregroundStyle(Theme.textPrimary)
                                        .textSelection(.enabled)
                                        .lineLimit(3)
                                    Spacer()
                                    Button {
                                        UIPasteboard.general.string = redirectUri
                                    } label: {
                                        Image(systemName: "doc.on.doc")
                                    }
                                    .buttonStyle(.plain)
                                    .foregroundStyle(Theme.accent)
                                    .accessibilityLabel("Copy redirect address")
                                }
                                .padding(8)
                                .background(RoundedRectangle(cornerRadius: 8).fill(Theme.bg))
                            }
                        }
                        .padding(.vertical, 2)
                        .task(id: p.type) { await loadRedirectUri() }
                    } else {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Run this on a computer with rclone installed, approve the sign-in, then paste what it prints below. No app registration needed — it uses rclone's own.")
                                .font(.caption)
                                .foregroundStyle(Theme.textSecondary)
                            HStack {
                                Text(p.authorizeCommand)
                                    .font(.caption.monospaced())
                                    .foregroundStyle(Theme.textPrimary)
                                    .textSelection(.enabled)
                                Spacer()
                                Button {
                                    UIPasteboard.general.string = p.authorizeCommand
                                } label: {
                                    Image(systemName: "doc.on.doc")
                                }
                                .buttonStyle(.plain)
                                .foregroundStyle(Theme.accent)
                                .accessibilityLabel("Copy command")
                            }
                            .padding(8)
                            .background(RoundedRectangle(cornerRadius: 8).fill(Theme.bg))
                        }
                        .padding(.vertical, 2)
                    }
                }

                ForEach(p.fields.filter { field in
                    browserAuth
                        ? (field.key == "client_id" || field.key == "client_secret")
                        : (field.key != "client_id" && field.key != "client_secret")
                }) { field in
                    providerField(field)
                }

                if browserAuth {
                    Button {
                        Task { await startBrowserSignIn() }
                    } label: {
                        HStack {
                            Spacer()
                            if signingIn {
                                ProgressView().tint(Theme.accent)
                            } else {
                                Text("Sign in to \(p.label)")
                                    .foregroundStyle(canBrowserSignIn ? Theme.accent : Theme.textSecondary)
                            }
                            Spacer()
                        }
                    }
                    .disabled(!canBrowserSignIn)
                } else {
                    Button {
                        Task { await createRemote() }
                    } label: {
                        HStack {
                            Spacer()
                            if creating {
                                ProgressView().tint(Theme.accent)
                            } else {
                                Text("Connect")
                                    .foregroundStyle(canCreateRemote ? Theme.accent : Theme.textSecondary)
                            }
                            Spacer()
                        }
                    }
                    .disabled(!canCreateRemote)
                }
            }

            if let setupMessage {
                Text(setupMessage)
                    .font(.footnote)
                    .foregroundStyle(setupOK ? Theme.success : Theme.danger)
            }
        } header: {
            Text("Add storage")
        } footer: {
            Text("Saves the account and immediately checks that it works.")
        }
        .listRowBackground(Theme.surface)
    }

    @ViewBuilder
    private func providerField(_ field: RcloneField) -> some View {
        let binding = Binding<String>(
            get: { newValues[field.key] ?? field.default },
            set: { newValues[field.key] = $0 }
        )
        VStack(alignment: .leading, spacing: 4) {
            if field.kind == "select" {
                Picker(field.label, selection: binding) {
                    ForEach(field.options, id: \.self) { Text($0).tag($0) }
                }
                .tint(Theme.accent)
            } else {
                Text(field.label + (field.required ? "" : " (optional)"))
                    .foregroundStyle(Theme.textPrimary)
                if field.kind == "secret" {
                    SecureField(field.placeholder.isEmpty ? field.label : field.placeholder,
                                text: binding)
                        .foregroundStyle(Theme.textPrimary)
                } else {
                    // token fields are long JSON blobs; axis .vertical lets the
                    // paste wrap instead of scrolling off one line.
                    TextField(
                        field.placeholder.isEmpty ? field.label : field.placeholder,
                        text: binding,
                        axis: field.kind == "token" ? .vertical : .horizontal
                    )
                    .lineLimit(field.kind == "token" ? 2...5 : 1...1)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .foregroundStyle(Theme.textPrimary)
                }
            }
            if !field.help.isEmpty {
                Text(field.help)
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
            }
        }
        .padding(.vertical, 2)
    }

    private func loadRcloneCatalogue() async {
        guard let api = session.api else { return }
        providers = (try? await api.rcloneProviders()) ?? []
        if let res = try? await api.rcloneRemotes() {
            rcloneAvailable = res.available
            remotes = res.remotes
        } else {
            // A backend predating the feature 404s — reported as unavailable so
            // the section explains the rebuild rather than showing a bare error.
            rcloneAvailable = false
            remotes = []
        }
    }

    private func createRemote() async {
        guard let api = session.api, let p = selectedProvider, !creating else { return }
        creating = true
        defer { creating = false }
        setupMessage = nil
        do {
            let res = try await api.createRcloneRemote(
                name: newName.trimmed, type: p.type, values: newValues
            )
            if !res.ok {
                setupOK = false
                setupMessage = res.detail.isEmpty ? "Could not save the account." : res.detail
            } else if !res.reachable {
                // Saved-but-unreachable is its OWN outcome. Calling it success
                // hides a wrong token; calling it a save failure makes the
                // operator re-enter something that stored fine.
                setupOK = false
                setupMessage = "Saved, but it did not answer: \(res.detail)"
            } else {
                setupOK = true
                setupMessage = "Connected. Use \(res.suggestedRemote) below."
                if archiveRemote.trimmed.isEmpty { archiveRemote = res.suggestedRemote }
                newValues = [:]
            }
            await loadRcloneCatalogue()
        } catch {
            session.handleAPIError(error)
            setupOK = false
            setupMessage = (error as? ApiError)?.message ?? error.localizedDescription
        }
    }

    private func testRemote(_ name: String) async {
        guard let api = session.api, busyRemote == nil else { return }
        busyRemote = name
        defer { busyRemote = nil }
        setupMessage = nil
        do {
            let res = try await api.testRcloneRemote(name: name)
            setupOK = res.ok
            if res.ok {
                let folders = res.folders.prefix(5).joined(separator: ", ")
                setupMessage = folders.isEmpty
                    ? "\(name): connected (no folders yet)"
                    : "\(name): connected — \(folders)"
            } else {
                setupMessage = "\(name): \(res.detail)"
            }
        } catch {
            session.handleAPIError(error)
            setupOK = false
            setupMessage = (error as? ApiError)?.message ?? error.localizedDescription
        }
    }

    private func removeRemote(_ name: String) async {
        guard let api = session.api, busyRemote == nil else { return }
        busyRemote = name
        defer { busyRemote = nil }
        do {
            try await api.deleteRcloneRemote(name: name)
            await loadRcloneCatalogue()
        } catch {
            session.handleAPIError(error)
            setupOK = false
            setupMessage = (error as? ApiError)?.message ?? error.localizedDescription
        }
    }


    /// The address this app reaches the server on — the redirect URI is derived
    /// from it, and must match what is registered on the provider's app.
    private var serverOrigin: String {
        guard let base = session.api?.apiBase,
              let scheme = base.scheme, let host = base.host
        else { return "" }
        let port = base.port.map { ":\($0)" } ?? ""
        return "\(scheme)://\(host)\(port)"
    }

    private var browserAuth: Bool {
        (selectedProvider?.oauth ?? false) && authMode == .browser
    }

    private var canBrowserSignIn: Bool {
        guard selectedProvider != nil, !newName.trimmed.isEmpty, !signingIn else { return false }
        return !(newValues["client_id"] ?? "").trimmed.isEmpty
            && !(newValues["client_secret"] ?? "").trimmed.isEmpty
    }

    private func loadRedirectUri() async {
        guard let api = session.api, selectedProvider?.oauth == true else { return }
        redirectUri = (try? await api.rcloneRedirectUri(origin: serverOrigin)) ?? ""
    }

    private func startBrowserSignIn() async {
        guard let api = session.api, let p = selectedProvider, !signingIn else { return }
        signingIn = true
        defer { signingIn = false }
        setupMessage = nil
        do {
            let res = try await api.startRcloneOAuth(
                name: newName.trimmed, type: p.type,
                clientId: newValues["client_id"] ?? "",
                clientSecret: newValues["client_secret"] ?? "",
                origin: serverOrigin
            )
            guard let url = URL(string: res.authUrl) else {
                setupOK = false
                setupMessage = "The provider returned an address this device cannot open."
                return
            }
            // Safari, not an in-app sheet: providers routinely refuse to sign in
            // inside an embedded web view, and the operator may already be
            // signed in on Safari.
            await UIApplication.shared.open(url)
            setupOK = true
            setupMessage = "Approve the sign-in in Safari, then pull down here to refresh."
        } catch {
            session.handleAPIError(error)
            setupOK = false
            setupMessage = (error as? ApiError)?.message ?? error.localizedDescription
        }
    }

    // MARK: - Cloud archive

    /// Mirrors the web's Integrations → "Cloud archive" card. A DIFFERENT
    /// settings subtree from MQTT above, so `save()` patches each only when it
    /// changed — editing the broker must never rewrite archive settings.
    private var archiveSection: some View {
        Section {
            Toggle("Upload event media nightly", isOn: $archiveEnabled)
                .tint(Theme.accent)

            VStack(alignment: .leading, spacing: 4) {
                Text("Remote").foregroundStyle(Theme.textPrimary)
                TextField("dropbox:Vigilume", text: $archiveRemote)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .foregroundStyle(Theme.textPrimary)
                Text("The rclone remote and path, as name:path. Set it up once on the server with `rclone config`.")
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
            }
            .padding(.vertical, 2)

            Stepper(value: $archiveHour, in: 0 ... 23) {
                HStack {
                    Text("Run at").foregroundStyle(Theme.textPrimary)
                    Spacer()
                    Text(String(format: "%02d:00", archiveHour))
                        .font(.subheadline.weight(.medium).monospacedDigit())
                        .foregroundStyle(Theme.textPrimary)
                }
            }
            .tint(Theme.accent)

            VStack(alignment: .leading, spacing: 4) {
                Stepper(value: $archiveKeepDays, in: 0 ... 3650, step: 5) {
                    HStack {
                        Text("Keep in cloud").foregroundStyle(Theme.textPrimary)
                        Spacer()
                        Text(archiveKeepDays == 0 ? "Forever" : "\(archiveKeepDays) days")
                            .font(.subheadline.weight(.medium).monospacedDigit())
                            .foregroundStyle(archiveKeepDays == 0 ? Theme.warning : Theme.textPrimary)
                    }
                }
                .tint(Theme.accent)
                Text(archiveKeepDays == 0
                     ? "Nothing is ever deleted from the cloud — the archive grows without limit."
                     : "Older day folders are deleted from the cloud. Independent of local retention.")
                    .font(.caption)
                    .foregroundStyle(archiveKeepDays == 0 ? Theme.warning : Theme.textSecondary)
            }
            .padding(.vertical, 2)

            Toggle("Include snapshots", isOn: $archiveIncludeSnapshots)
                .tint(Theme.accent)

            VStack(alignment: .leading, spacing: 4) {
                Text("Upload speed limit").foregroundStyle(Theme.textPrimary)
                TextField("unlimited", text: $archiveBwlimit)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .foregroundStyle(Theme.textPrimary)
                Text("e.g. 2M. Worth setting on a thin uplink so the nightly run does not starve live view.")
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
            }
            .padding(.vertical, 2)
        } header: {
            Text("Cloud archive")
        } footer: {
            Text("Copies each finished day of event clips and snapshots to cloud storage overnight, one folder per day. 24/7 footage is never uploaded — far too large for any normal connection. This backs up the evidence, not everything.")
        }
        .listRowBackground(Theme.surface)
    }

    private var archiveStatusSection: some View {
        Section {
            Button {
                Task { await runArchiveNow() }
            } label: {
                HStack {
                    Spacer()
                    if archiveRunning {
                        ProgressView().tint(Theme.accent)
                    } else {
                        Text("Run now")
                            .foregroundStyle(canRunArchive ? Theme.accent : Theme.textSecondary)
                    }
                    Spacer()
                }
            }
            .disabled(!canRunArchive)

            if let archiveError {
                Text(archiveError)
                    .font(.footnote)
                    .foregroundStyle(Theme.danger)
            }
            if let status = archiveStatus, status.available {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Archived through: \(status.lastUploadedDay ?? "nothing yet")")
                        .font(.footnote)
                        .foregroundStyle(Theme.textPrimary)
                    if let at = status.lastResult.at {
                        Text(lastRunSummary(status.lastResult, at: at))
                            .font(.caption)
                            .foregroundStyle(Theme.textSecondary)
                    }
                    if let errors = status.lastResult.errors, !errors.isEmpty {
                        Text(errors.joined(separator: " · "))
                            .font(.caption)
                            .foregroundStyle(Theme.danger)
                    }
                }
                .padding(.vertical, 2)
            }
        } footer: {
            Text("Runs the real nightly pass, so a clean result proves the remote works. It uses the SAVED settings — save first. A day of clips can take a while on a slow connection.")
        }
        .listRowBackground(Theme.surface)
    }

    /// Gated on the SAVED state, not the draft: the run uses what the server
    /// holds, so offering the button for unsaved settings would test the wrong
    /// thing and report a confusing result.
    private var canRunArchive: Bool {
        guard let a = doc?.archive else { return false }
        return a.enabled && !a.remote.isEmpty && !archiveRunning && !saving
    }

    private func lastRunSummary(_ r: ArchiveStatus.RunResult, at: String) -> String {
        let uploaded = r.uploadedDays?.joined(separator: ", ")
        var parts = ["Last run \(at)"]
        parts.append("uploaded \(uploaded?.isEmpty == false ? uploaded! : "nothing")")
        if let files = r.files, files > 0 { parts.append("\(files) files") }
        if let pruned = r.prunedDays, !pruned.isEmpty {
            parts.append("expired \(pruned.joined(separator: ", "))")
        }
        return parts.joined(separator: " — ")
    }

    private func runArchiveNow() async {
        guard let api = session.api, !archiveRunning else { return }
        archiveRunning = true
        defer { archiveRunning = false }
        archiveError = nil
        do {
            let result = try await api.runArchive()
            if !result.ok, !result.detail.isEmpty { archiveError = result.detail }
            archiveStatus = try? await api.archiveStatus()
        } catch {
            session.handleAPIError(error)
            archiveError = (error as? ApiError)?.message ?? error.localizedDescription
        }
    }

    private func loadArchiveStatus() async {
        guard let api = session.api else { return }
        // A backend predating the archive 404s here; the card's own copy already
        // explains what is needed, so a failure just leaves the status hidden.
        archiveStatus = try? await api.archiveStatus()
    }

    // MARK: - Load / Test / Save

    private func load() async {
        guard session.isAdmin, let api = session.api else { return }
        loading = doc == nil
        do {
            apply(try await api.settingsDocument())
            loadError = nil
        } catch {
            session.handleAPIError(error)
            loadError = (error as? ApiError)?.message ?? error.localizedDescription
        }
        loading = false
    }

    private func test() async {
        guard let api = session.api, !testing else { return }
        testing = true
        defer { testing = false }
        testResult = nil
        do {
            testResult = try await api.testMqtt(
                enabled: enabled,
                host: host.trimmed,
                port: clampedPort,
                username: username.trimmed,
                // The DRAFT password: what the user typed, or empty to let the
                // backend probe with the stored secret.
                password: password,
                discoveryPrefix: discoveryPrefix.trimmed,
                baseTopic: baseTopic.trimmed
            )
        } catch {
            session.handleAPIError(error)
            // Surface a transport/HTTP failure in the same red line as a broker
            // refusal rather than dropping it silently.
            testResult = MqttTestResult(
                ok: false,
                detail: (error as? ApiError)?.message ?? error.localizedDescription
            )
        }
    }

    private func save() async {
        guard let api = session.api, !saving else { return }
        saving = true
        defer { saving = false }
        saveError = nil
        // Every field on SettingsPatch.Mqtt is Optional and JSONEncoder omits
        // nil, so the backend deep-merge only touches what's sent. The whole
        // block rides one PATCH (like ntfy), with the one twist below.
        // Each subtree is sent ONLY when it changed. Both ride one PATCH, and
        // the backend deep-merges, so an untouched block is never rewritten.
        let patch = SettingsPatch(
            mqtt: mqttDirty ? .init(
                enabled: enabled,
                host: host.trimmed,
                port: clampedPort,
                username: username.trimmed,
                // SECRET CARRY-FORWARD: the server masks `password`, so the
                // field loaded EMPTY. Empty => send nil and the deep-merge keeps
                // the stored secret; non-empty => a deliberate replacement.
                // Identical in shape to how PhonePushSettingsView guards
                // `ntfy.authToken` from being clobbered by a masked value.
                password: passwordChanged ? password : nil,
                discoveryPrefix: discoveryPrefix.trimmed,
                baseTopic: baseTopic.trimmed
            ) : nil,
            archive: archiveDirty ? .init(
                enabled: archiveEnabled,
                remote: archiveRemote.trimmed,
                hour: archiveHour,
                keepDays: archiveKeepDays,
                includeSnapshots: archiveIncludeSnapshots,
                bwlimit: archiveBwlimit.trimmed
            ) : nil
        )
        do {
            apply(try await api.patchSettings(patch))
        } catch {
            session.handleAPIError(error)
            // A bad host/topic/port is a readable 422 from the backend's
            // validators — show it verbatim.
            saveError = (error as? ApiError)?.message ?? error.localizedDescription
        }
    }

    /// Adopt a server document (initial load or a PATCH response) as both the
    /// baseline and the draft — the response is the full updated document, so a
    /// save leaves the form showing exactly what the server now holds.
    private func apply(_ document: SettingsDocument) {
        doc = document
        let m = document.mqtt
        enabled = m.enabled
        host = m.host
        portText = String(m.port)
        username = m.username
        discoveryPrefix = m.discoveryPrefix
        baseTopic = m.baseTopic
        let a = document.archive
        archiveEnabled = a.enabled
        archiveRemote = a.remote
        archiveHour = a.hour
        archiveKeepDays = a.keepDays
        archiveIncludeSnapshots = a.includeSnapshots
        archiveBwlimit = a.bwlimit
        // Deliberately NOT `password = m.password`: the server sends it masked,
        // and pre-filling the SecureField would let that placeholder be saved
        // back as the real secret. Empty field = "keep the stored password".
        password = ""
        // A prior probe no longer reflects the reloaded config.
        testResult = nil
    }
}

private extension String {
    var trimmed: String { trimmingCharacters(in: .whitespacesAndNewlines) }
}
