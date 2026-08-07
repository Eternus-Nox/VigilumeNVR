import SwiftUI

/// Sign in to another Vigilume server. On success SessionModel.login adds the
/// server to the store and makes it active — the whole app switches to it
/// (the multi-server self-host story). Mirrors LoginView's field conventions.
struct AddServerSheet: View {
    @EnvironmentObject private var session: SessionModel
    @Environment(\.dismiss) private var dismiss

    @State private var name = ""
    @State private var serverURL = ""
    @State private var lanURL = ""
    @State private var username = "admin"
    @State private var password = ""
    @State private var errorMessage: String?
    @State private var isConnecting = false

    private var normalizedURL: String {
        ServerConfig.normalizeURLString(serverURL)
    }

    private var isInsecureHTTP: Bool {
        normalizedURL.lowercased().hasPrefix("http://")
    }

    private var canSubmit: Bool {
        !normalizedURL.isEmpty && !username.isEmpty && !password.isEmpty && !isConnecting
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("Name (optional)", text: $name)
                    TextField("https://nvr.example.com", text: $serverURL)
                        .textContentType(.URL)
                        .keyboardType(.URL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                } header: {
                    Text("External address (IP or domain)")
                } footer: {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("The address that works from ANYWHERE — your domain or public "
                             + "IP. Used for everything when you're away from home.")
                        if isInsecureHTTP, !serverURL.isEmpty {
                            Label("Not encrypted — credentials travel in cleartext over http",
                                  systemImage: "lock.open.fill")
                                .foregroundStyle(Theme.warning)
                        }
                    }
                }
                .listRowBackground(Theme.surface)

                Section {
                    TextField("http://192.168.1.253:8080", text: $lanURL)
                        .textContentType(.URL)
                        .keyboardType(.URL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                } header: {
                    Text("Internal address (local network)")
                } footer: {
                    Text("Your NVR's address on the home network. Video is pulled from here "
                         + "whenever it's reachable — that's what makes live view fast, and "
                         + "it's the only path WebRTC can use. A VPN address (WireGuard or "
                         + "Tailscale) works here too, giving full-speed live view from "
                         + "anywhere with nothing exposed to the internet. Leave blank to "
                         + "always use the external address above.")
                }
                .listRowBackground(Theme.surface)

                Section("Sign In") {
                    TextField("Username", text: $username)
                        .textContentType(.username)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                    SecureField("Password", text: $password)
                        .textContentType(.password)
                }
                .listRowBackground(Theme.surface)

                if let errorMessage {
                    Section {
                        Text(errorMessage)
                            .font(.callout)
                            .foregroundStyle(Theme.danger)
                    }
                    .listRowBackground(Theme.surface)
                }

                Section {
                    Button(action: submit) {
                        HStack {
                            Spacer()
                            if isConnecting {
                                ProgressView()
                            } else {
                                Text("Connect")
                                    .fontWeight(.semibold)
                            }
                            Spacer()
                        }
                    }
                    .disabled(!canSubmit)
                }
                .listRowBackground(Theme.surface)
            }
            .scrollContentBackground(.hidden)
            .background(Theme.bg)
            .navigationTitle("Add Server")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
            }
        }
    }

    private func submit() {
        guard canSubmit else { return }
        errorMessage = nil
        isConnecting = true
        let serverName = name
        let url = normalizedURL
        let lan = lanURL
        let user = username
        let pass = password
        Task {
            defer { isConnecting = false }
            do {
                try await session.login(
                    serverName: serverName, urlString: url, lanURLString: lan,
                    username: user, password: pass
                )
                PushManager.shared.syncToActiveServer()
                dismiss()
            } catch let error as ApiError {
                errorMessage = error.status == 401
                    ? "Wrong username or password"
                    : error.message
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }
}
