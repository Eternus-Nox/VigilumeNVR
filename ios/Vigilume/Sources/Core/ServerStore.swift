import Foundation
import Combine

/// One saved NVR server. Name + URL live in UserDefaults; the JWT lives in
/// the Keychain keyed by the server's id.
struct ServerConfig: Codable, Identifiable, Equatable, Sendable {
    let id: UUID
    var name: String
    var urlString: String
    /// Optional LAN-only address (e.g. `http://192.168.1.50:8080`) used ONLY for
    /// fast, direct video/media when the device is on a network where it's
    /// reachable (home Wi-Fi). Control/auth/list calls always ride `urlString`.
    /// Absent (nil) on servers saved before this feature — migration is a no-op
    /// because the synthesized Codable decodes a missing key as nil, i.e.
    /// "everything on the primary URL, exactly as today".
    var lanURLString: String?

    init(
        id: UUID = UUID(),
        name: String,
        urlString: String,
        lanURLString: String? = nil
    ) {
        self.id = id
        self.name = name
        self.urlString = urlString
        self.lanURLString = lanURLString
    }

    // MARK: Decoding

    /// Written out by hand, and it MUST stay that way.
    ///
    /// Swift's SYNTHESIZED `Decodable` does not use property default values: for
    /// a non-optional stored property it emits `decode(_:forKey:)`, which THROWS
    /// `.keyNotFound` when older saved JSON lacks the key. `ServerStore.init`
    /// decodes the whole array with `try?`, so a single missing key on a single
    /// server does not skip that server — it wipes EVERY saved server and
    /// strands their Keychain JWTs, because `removeServer` never runs to clean
    /// them up. Adding one non-optional field with a default is all it takes.
    ///
    /// (Optional properties are safe either way — the synthesized decoder uses
    /// `decodeIfPresent` for those, which is why `lanURLString` could be added
    /// without this. A now-removed `directURLString` key in older saved JSON is
    /// simply ignored, since the decoder does not error on unknown keys.)
    ///
    /// So: `id` is the only required key. Everything else tolerates absence and
    /// falls back to the same default the memberwise init uses. Adding a field
    /// here is then always a no-op migration, whatever its type.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(UUID.self, forKey: .id)
        name = try c.decodeIfPresent(String.self, forKey: .name) ?? ""
        urlString = try c.decodeIfPresent(String.self, forKey: .urlString) ?? ""
        lanURLString = try c.decodeIfPresent(String.self, forKey: .lanURLString)
    }

    var url: URL? { URL(string: urlString) }

    /// The parsed LAN URL, or nil when none is configured / it's blank.
    var lanURL: URL? {
        guard let s = lanURLString, !s.isEmpty else { return nil }
        return URL(string: s)
    }

    /// Media routes to try, IN PREFERENCE ORDER.
    ///
    /// LAN/VPN first: when it answers it is the fastest, fully-encrypted path
    /// and needs nothing exposed to the internet. Anything not listed here
    /// falls back to the primary URL, which always works. Put your WireGuard
    /// (or Wi-Fi LAN) address in `lanURLString` for full-speed video from
    /// anywhere; with none set, all media rides the primary URL.
    func mediaRoutes() -> [MediaRoute] {
        var routes: [MediaRoute] = []
        if let lanURL { routes.append(.lan(lanURL)) }
        return routes
    }

    /// True when the base URL is plain http — the UI shows a
    /// "not encrypted" badge for these (ios-design.md §5 ATS decision).
    var isInsecureHTTP: Bool {
        url?.scheme?.lowercased() == "http"
    }

    /// Normalize user input: add a scheme if missing, strip trailing slash.
    static func normalizeURLString(_ raw: String) -> String {
        var s = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if !s.isEmpty, !s.contains("://") { s = "http://\(s)" }
        while s.hasSuffix("/") { s.removeLast() }
        return s
    }

    /// Normalize an optional LAN field: blank input becomes nil (no LAN route).
    static func normalizeOptionalURLString(_ raw: String?) -> String? {
        guard let raw else { return nil }
        let normalized = normalizeURLString(raw)
        return normalized.isEmpty ? nil : normalized
    }
}

/// Multi-server persistence + active-server switching.
/// UserDefaults: server list + active id. Keychain: one JWT per server.
@MainActor
final class ServerStore: ObservableObject {
    private static let serversKey = "vigilume.servers"
    private static let activeKey = "vigilume.activeServerID"
    /// Pre-rename keys (the app shipped as "Sentinel"). Read once, copied
    /// forward, then removed — see `migrateLegacyKeys`.
    private static let legacyServersKey = "sentinel.servers"
    private static let legacyActiveKey = "sentinel.activeServerID"
    private static func tokenKey(_ id: UUID) -> String { "jwt.\(id.uuidString)" }

    /// One-time move of the saved-server list + active id off the pre-rename
    /// UserDefaults keys. Without it the rename silently empties the server
    /// list, which reads as "the app forgot every server I ever added".
    ///
    /// Idempotent: each key is copied only when the new one is absent, and the
    /// old key is removed after the copy, so a second run finds nothing to do.
    /// Missing legacy values are simply skipped — a fresh install migrates
    /// nothing and no read can fail.
    private static func migrateLegacyKeys(in defaults: UserDefaults) {
        if defaults.object(forKey: serversKey) == nil,
           let legacy = defaults.data(forKey: legacyServersKey) {
            defaults.set(legacy, forKey: serversKey)
        }
        if defaults.object(forKey: activeKey) == nil,
           let legacy = defaults.string(forKey: legacyActiveKey) {
            defaults.set(legacy, forKey: activeKey)
        }
        defaults.removeObject(forKey: legacyServersKey)
        defaults.removeObject(forKey: legacyActiveKey)
    }

    /// Decodes one server, turning a failure into nil instead of poisoning the
    /// whole array. See the note in `init` on why per-element matters.
    private struct FailableServer: Decodable {
        let value: ServerConfig?
        init(from decoder: Decoder) throws {
            value = try? ServerConfig(from: decoder)
        }
    }

    @Published private(set) var servers: [ServerConfig]
    @Published private(set) var activeServerID: UUID?

    private let defaults: UserDefaults

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        Self.migrateLegacyKeys(in: defaults)
        // Per-element, NOT `decode([ServerConfig].self)`. Decoding the array as
        // one unit means any single unreadable row throws and `try?` silently
        // yields nil — i.e. one bad server logs you out of all of them. Losing
        // saved servers is close to the worst non-destructive failure this app
        // has, so it degrades one row at a time instead.
        if let data = defaults.data(forKey: Self.serversKey),
           let rows = try? JSONDecoder().decode([FailableServer].self, from: data) {
            servers = rows.compactMap(\.value)
        } else {
            servers = []
        }
        if let raw = defaults.string(forKey: Self.activeKey), let id = UUID(uuidString: raw) {
            activeServerID = id
        }
        // Heal a dangling active id.
        if let id = activeServerID, !servers.contains(where: { $0.id == id }) {
            activeServerID = servers.first?.id
        }
    }

    var activeServer: ServerConfig? {
        guard let id = activeServerID else { return nil }
        return servers.first { $0.id == id }
    }

    // MARK: CRUD

    /// Add (or update the URL/LAN of an existing same-URL entry) and make active.
    @discardableResult
    func addServer(
        name: String,
        urlString: String,
        lanURLString: String? = nil
    ) -> ServerConfig {
        let normalized = ServerConfig.normalizeURLString(urlString)
        let normalizedLAN = ServerConfig.normalizeOptionalURLString(lanURLString)
        if let idx = servers.firstIndex(where: { $0.urlString == normalized }) {
            // Re-signing into a known server: keep it, but adopt a newly
            // supplied LAN address (an empty field leaves the existing one).
            if let normalizedLAN { servers[idx].lanURLString = normalizedLAN }
            setActive(servers[idx].id)
            persist()
            return servers[idx]
        }
        let displayName = name.trimmingCharacters(in: .whitespaces)
        let server = ServerConfig(
            name: displayName.isEmpty ? normalized : displayName,
            urlString: normalized,
            lanURLString: normalizedLAN
        )
        servers.append(server)
        setActive(server.id)
        persist()
        return server
    }

    func updateServer(_ server: ServerConfig) {
        guard let idx = servers.firstIndex(where: { $0.id == server.id }) else { return }
        servers[idx] = server
        persist()
    }

    func removeServer(id: UUID) {
        KeychainHelper.remove(forKey: Self.tokenKey(id))
        // Forget the server's E2E push key too — the extension iterates all
        // stored keys per push, so stale ones are pure overhead.
        PushCrypto.removeKey(forServer: id.uuidString)
        servers.removeAll { $0.id == id }
        if activeServerID == id {
            activeServerID = servers.first?.id
            persistActive()
        }
        persist()
    }

    func setActive(_ id: UUID) {
        guard servers.contains(where: { $0.id == id }) else { return }
        activeServerID = id
        persistActive()
    }

    // MARK: Tokens (Keychain)

    func token(for id: UUID) -> String? {
        KeychainHelper.string(forKey: Self.tokenKey(id))
    }

    var activeToken: String? {
        guard let id = activeServerID else { return nil }
        return token(for: id)
    }

    func setToken(_ token: String, for id: UUID) {
        KeychainHelper.setString(token, forKey: Self.tokenKey(id))
    }

    func clearToken(for id: UUID) {
        KeychainHelper.remove(forKey: Self.tokenKey(id))
    }

    // MARK: Persistence

    private func persist() {
        if let data = try? JSONEncoder().encode(servers) {
            defaults.set(data, forKey: Self.serversKey)
        }
    }

    private func persistActive() {
        defaults.set(activeServerID?.uuidString, forKey: Self.activeKey)
    }
}
