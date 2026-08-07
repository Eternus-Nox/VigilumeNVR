import SwiftUI

/// Edit where VIDEO comes from for a saved server.
///
/// Only media is routed. Login, the camera list, events, settings and both
/// WebSockets always ride the primary URL, so a wrong entry here can never lock
/// anyone out — the worst case is that video falls back to the primary URL,
/// which is exactly today's behaviour.
///
/// WHY THIS SCREEN EXISTS. The primary URL is usually fronted by a CDN tunnel.
/// That carries the control API perfectly but is a poor pipe for sustained
/// video — segment fetches get buffered and throttled, and most CDN terms
/// restrict serving video at all — which is what makes remote live view stutter
/// while the rest of the app feels fine. Pointing media at a LAN/VPN address
/// fixes live view without giving up the tunnel for anything else.
struct VideoRoutingSheet: View {
    let server: ServerConfig

    @EnvironmentObject private var serverStore: ServerStore
    @Environment(\.dismiss) private var dismiss

    @State private var lanURL: String
    @State private var errorMessage: String?

    /// Held as a static rather than inline: a long `+`-concatenated literal
    /// inside a ViewBuilder is what makes Swift's type-checker give up.
    private static let lanFooter = """
        On your home network this is your NVR's LAN address. If you run a VPN, \
        put that same address here — your router has WireGuard built in, and \
        with it on, this path works at full speed from anywhere, encrypted, \
        with nothing exposed to the internet. Leave it blank to send all video \
        over the primary address.
        """

    init(server: ServerConfig) {
        self.server = server
        _lanURL = State(initialValue: server.lanURLString ?? "")
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("http://192.168.1.253:8080", text: $lanURL)
                        .textContentType(.URL)
                        .keyboardType(.URL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                } header: {
                    Text("Internal address (local network)")
                } footer: {
                    Text(Self.lanFooter)
                }
                .listRowBackground(Theme.surface)

                Section {
                    Label("Video only", systemImage: "info.circle")
                        .font(.caption)
                        .foregroundStyle(Theme.textSecondary)
                    Text("Everything else always uses \(server.urlString). If the "
                         + "address above can't be reached, video quietly falls "
                         + "back to it too.")
                        .font(.caption)
                        .foregroundStyle(Theme.textSecondary)
                } .listRowBackground(Theme.surface)

                if let errorMessage {
                    Section {
                        Text(errorMessage).foregroundStyle(Theme.danger).font(.caption)
                    } .listRowBackground(Theme.surface)
                }
            }
            .scrollContentBackground(.hidden)
            .background(Theme.bg)
            .navigationTitle("Video routing")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") { save() }
                }
            }
        }
    }

    private func save() {
        // Validate exactly like the primary URL: a malformed entry is rejected
        // rather than silently stored and then silently unreachable.
        guard let lan = validated(lanURL, label: "local / VPN") else { return }

        var updated = server
        updated.lanURLString = lan
        serverStore.updateServer(updated)

        // Re-probe immediately so the change takes effect now rather than on the
        // next network change — and so the badge tells the truth straight away.
        if updated.id == serverStore.activeServerID {
            LANReachability.shared.setActiveServer(updated)
        }
        dismiss()
    }

    /// nil-on-blank, validated otherwise. Blank simply yields nil (clears the
    /// field) and sets no error; a malformed entry sets an error and returns nil
    /// so `save()` bails.
    private func validated(_ raw: String, label: String) -> String?? {
        guard let normalized = ServerConfig.normalizeOptionalURLString(raw) else {
            return .some(nil)   // blank — clear the field
        }
        guard let url = URL(string: normalized), url.host != nil,
              url.scheme == "http" || url.scheme == "https"
        else {
            errorMessage = "Enter a valid \(label) address, e.g. http://192.168.1.50:8080"
            return nil
        }
        return .some(normalized)
    }
}
