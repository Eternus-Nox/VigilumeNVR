import SwiftUI

struct LoginView: View {
    @EnvironmentObject private var session: SessionModel
    @EnvironmentObject private var serverStore: ServerStore

    @State private var serverURL: String = ""
    @State private var lanURL: String = ""
    @State private var username: String = "admin"
    @State private var password: String = ""
    @State private var errorMessage: String?
    @State private var isLoggingIn = false

    private var normalizedURL: String {
        ServerConfig.normalizeURLString(serverURL)
    }

    private var isInsecureHTTP: Bool {
        normalizedURL.lowercased().hasPrefix("http://")
    }

    private var canSubmit: Bool {
        !normalizedURL.isEmpty && !username.isEmpty && !password.isEmpty && !isLoggingIn
    }

    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                header

                VStack(spacing: 14) {
                    field("Server URL", text: $serverURL,
                          placeholder: "http://192.168.1.50:8080",
                          contentType: .URL, keyboard: .URL)

                    if isInsecureHTTP, !serverURL.isEmpty {
                        Label("Not encrypted — credentials travel in cleartext over http",
                              systemImage: "lock.open.fill")
                            .font(.footnote)
                            .foregroundStyle(Theme.warning)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }

                    field("Local network address (LAN) — optional", text: $lanURL,
                          placeholder: "http://192.168.1.50:8080",
                          contentType: .URL, keyboard: .URL)
                    Text("Used for fast local video when you're on this network; leave blank to always use the main address.")
                        .font(.caption)
                        .foregroundStyle(Theme.textSecondary)
                        .frame(maxWidth: .infinity, alignment: .leading)

                    field("Username", text: $username,
                          placeholder: "admin", contentType: .username)

                    secureField("Password", text: $password)
                }

                if let errorMessage {
                    Text(errorMessage)
                        .font(.callout)
                        .foregroundStyle(Theme.danger)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }

                Button(action: submit) {
                    Group {
                        if isLoggingIn {
                            ProgressView().tint(Theme.bg)
                        } else {
                            Text("Sign In").fontWeight(.semibold)
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
                }
                .background(canSubmit ? Theme.accent : Theme.borderStrong)
                .foregroundStyle(Theme.bg)
                .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                .disabled(!canSubmit)
            }
            .padding(24)
            .frame(maxWidth: 480)
        }
        .scrollBounceBehavior(.basedOnSize)
        .background(Theme.bg)
        .onAppear {
            // Prefill from the active saved server, if any.
            if serverURL.isEmpty, let server = serverStore.activeServer {
                serverURL = server.urlString
                if lanURL.isEmpty, let lan = server.lanURLString { lanURL = lan }
            }
            if let sessionError = session.lastError {
                errorMessage = sessionError
                session.lastError = nil
            }
        }
    }

    private var header: some View {
        VStack(spacing: 8) {
            Image(systemName: "shield.lefthalf.filled")
                .font(.system(size: 52))
                .foregroundStyle(Theme.accent)
            Text("Vigilume")
                .font(.largeTitle.bold())
                .foregroundStyle(Theme.textPrimary)
            Text("Self-hosted NVR")
                .font(.subheadline)
                .foregroundStyle(Theme.textSecondary)
        }
        .padding(.top, 48)
    }

    private func field(
        _ label: String,
        text: Binding<String>,
        placeholder: String,
        contentType: UITextContentType? = nil,
        keyboard: UIKeyboardType = .default
    ) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(label)
                .font(.footnote.weight(.medium))
                .foregroundStyle(Theme.textSecondary)
            TextField(placeholder, text: text)
                .textContentType(contentType)
                .keyboardType(keyboard)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .padding(12)
                .background(Theme.surfaceAlt)
                .foregroundStyle(Theme.textPrimary)
                .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .stroke(Theme.border, lineWidth: 1)
                )
        }
    }

    private func secureField(_ label: String, text: Binding<String>) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(label)
                .font(.footnote.weight(.medium))
                .foregroundStyle(Theme.textSecondary)
            SecureField("••••••••", text: text)
                .textContentType(.password)
                .padding(12)
                .background(Theme.surfaceAlt)
                .foregroundStyle(Theme.textPrimary)
                .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .stroke(Theme.border, lineWidth: 1)
                )
                .onSubmit { if canSubmit { submit() } }
        }
    }

    private func submit() {
        guard canSubmit else { return }
        errorMessage = nil
        isLoggingIn = true
        let url = normalizedURL
        let lan = lanURL
        let user = username
        let pass = password
        Task {
            defer { isLoggingIn = false }
            do {
                try await session.login(
                    serverName: "", urlString: url, lanURLString: lan,
                    username: user, password: pass
                )
                // Re-adopt this server's push preference (mirrors AddServerSheet).
                PushManager.shared.syncToActiveServer()
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
