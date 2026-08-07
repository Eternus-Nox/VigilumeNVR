import SwiftUI

/// Settings › Integrations: the iOS twin of the web's Settings → Integrations
/// "Home Assistant (MQTT)" card. Publishes camera + detection state to an MQTT
/// broker so Home Assistant can auto-discover this NVR's entities.
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
        .task { await load() }
        .refreshable { await load() }
    }

    private var form: some View {
        List {
            enableSection
            brokerSection
            authSection
            topicsSection
            testSection
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

    private var dirty: Bool {
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
        let patch = SettingsPatch(
            mqtt: .init(
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
            )
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
