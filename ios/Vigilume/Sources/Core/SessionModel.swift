import Foundation
import Combine

/// App-wide auth + live-update state. Owns the APIClient for the active
/// server and the /api/ws socket. Feature views reach it via
/// `@EnvironmentObject var session: SessionModel`.
@MainActor
final class SessionModel: ObservableObject {
    enum Phase: Equatable {
        case restoring      // launch: validating a stored token
        case loggedOut
        case loggedIn
    }

    @Published private(set) var phase: Phase = .restoring
    @Published private(set) var username: String?
    @Published private(set) var role: Role?
    /// Last auth/network error surfaced to the UI (LoginView reads this).
    @Published var lastError: String?

    /// Live camera online/offline map fed by WS camera_status frames.
    @Published private(set) var cameraOnline: [String: Bool] = [:]

    /// Deep-link target: vigilume://events/<id> parks the id here; the
    /// Events tab consumes and clears it.
    @Published var pendingEventID: Int?

    /// Answered-doorbell target: the CallKit answer parks the payload `camera`
    /// (friendly name, e.g. "Front Door") here; MainTabView switches to the
    /// Cameras tab and CamerasView opens that camera's live view, then clears it.
    @Published var pendingLiveCameraName: String?

    /// Re-published WS frames for feature views (event_new/update/end,
    /// doorbell, model_status...). Delivered on the main actor.
    let wsMessages = PassthroughSubject<WSMessage, Never>()
    /// Fires after a WS reconnect — refetch lists to close the gap.
    let wsReconnected = PassthroughSubject<Void, Never>()

    let serverStore: ServerStore
    let socket = EventSocket()
    /// Decides whether the active server's LAN address is reachable right now —
    /// drives the `mediaBase` of `api` (per-server LAN routing).
    let lanReachability = LANReachability.shared

    /// Client for the active server + token. Nil only when no server is
    /// configured. Rebuilt on login/logout/server switch — and whenever LAN
    /// reachability flips (to swap `mediaBase` between LAN and primary).
    /// `@Published` so live views re-resolve their media URLs on the next attach.
    @Published private(set) var api: APIClient?

    private var cancellables = Set<AnyCancellable>()

    var isAdmin: Bool { role == .admin }

    init(serverStore: ServerStore) {
        self.serverStore = serverStore
        // The route probe needs an API client for the optional public-IP
        // lookup, but must not own auth — hand it a getter instead.

        // LAN reachability flip ⇒ rebuild `api` with a new `mediaBase` so the
        // next live/media attach routes video to the LAN (or back to primary).
        // `receive(on:)` defers to the next runloop tick so `refreshMediaBase`
        // reads the UPDATED dictionary (@Published notifies in willSet).
        lanReachability.$routes
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in self?.refreshMediaBase() }
            .store(in: &cancellables)

        socket.messages
            .sink { [weak self] message in
                guard let self else { return }
                if case .cameraStatus(let updates) = message {
                    self.cameraOnline.merge(updates) { _, new in new }
                }
                self.wsMessages.send(message)
            }
            .store(in: &cancellables)

        socket.reconnected
            .sink { [weak self] in self?.wsReconnected.send() }
            .store(in: &cancellables)
    }

    // MARK: Launch restore

    /// Validate any stored session (GET /api/auth/me). Called once at launch.
    func restore() async {
        guard let server = serverStore.activeServer,
              let url = server.url,
              let token = serverStore.activeToken
        else {
            phase = .loggedOut
            return
        }
        // Kick off the LAN probe for this server; `mediaBase` starts on the
        // primary URL and swaps to LAN if/when the probe proves it reachable.
        lanReachability.setActiveServer(server)
        let client = makeAPIClient(token: token) ?? APIClient(apiBase: url, token: token)
        api = client
        do {
            let me = try await client.me()
            username = me.username
            role = me.role
            phase = .loggedIn
            startSocket()
        } catch let error as ApiError where error.isUnauthorized {
            serverStore.clearToken(for: server.id)
            api = APIClient(apiBase: url, token: nil)
            phase = .loggedOut
        } catch {
            // Server unreachable: keep the stored session, enter the app
            // offline; the first authed call that 401s will log out.
            username = cachedUsername(for: server.id)
            role = cachedRole(for: server.id)
            phase = .loggedIn
            startSocket()
        }
    }

    // MARK: Login / logout

    /// Add/select the server, POST /api/auth/login, persist the JWT in the
    /// Keychain, connect the WS. Throws ApiError with a UI-ready message.
    ///
    /// `lanURLString` is the OPTIONAL local-network address; when set + reachable
    /// the app routes video/media to it. Auth (this login) always uses the
    /// primary URL, so LAN is never on the credential path.
    func login(
        serverName: String, urlString: String, lanURLString: String? = nil,
        username: String, password: String
    ) async throws {
        let normalized = ServerConfig.normalizeURLString(urlString)
        guard let url = URL(string: normalized), url.host != nil,
              url.scheme == "http" || url.scheme == "https"
        else {
            throw ApiError(status: 0, message: "Enter a valid server URL, e.g. http://192.168.1.50:8080")
        }
        // Validate the LAN address like the primary (blank ⇒ no LAN route).
        let normalizedLAN = ServerConfig.normalizeOptionalURLString(lanURLString)
        if let normalizedLAN {
            guard let lan = URL(string: normalizedLAN), lan.host != nil,
                  lan.scheme == "http" || lan.scheme == "https"
            else {
                throw ApiError(status: 0, message: "Enter a valid local network address, e.g. http://192.168.1.50:8080")
            }
        }
        // Login always rides the primary URL — never the LAN.
        let client = APIClient(apiBase: url, token: nil)
        let response = try await client.login(username: username, password: password)

        let server = serverStore.addServer(
            name: serverName, urlString: normalized, lanURLString: normalizedLAN
        )
        serverStore.setToken(response.token, for: server.id)
        cacheIdentity(username: response.username, role: response.role, for: server.id)

        // Probe this server's media routes, then build the client (media moves
        // off the primary URL only once/if a route proves reachable).
        lanReachability.setActiveServer(server)
        api = makeAPIClient(token: response.token) ?? APIClient(apiBase: url, token: response.token)
        self.username = response.username
        role = response.role
        lastError = nil
        phase = .loggedIn
        startSocket()
    }

    /// Drop the active server's token and return to the login screen.
    func logout() {
        socket.disconnect()
        if let id = serverStore.activeServerID {
            serverStore.clearToken(for: id)
        }
        if let url = serverStore.activeServer?.url {
            api = APIClient(apiBase: url, token: nil)
        } else {
            api = nil
        }
        username = nil
        role = nil
        cameraOnline = [:]
        phase = .loggedOut
    }

    /// Switch to another saved server (re-validates its stored token).
    func switchServer(to id: UUID) async {
        socket.disconnect()
        serverStore.setActive(id)
        cameraOnline = [:]
        phase = .restoring
        await restore()
    }

    /// Feature views funnel request errors here: a 401 anywhere means the
    /// token died — return to login (mirrors the web client's redirect).
    func handleAPIError(_ error: Error) {
        if let apiError = error as? ApiError, apiError.isUnauthorized {
            lastError = "Session expired — please sign in again"
            logout()
        }
    }

    // MARK: Media base (per-server LAN routing)

    /// Build the active server's client: `apiBase` = primary URL (all control/
    /// auth), `mediaBase` = LAN when currently reachable else primary.
    private func makeAPIClient(token: String?) -> APIClient? {
        guard let server = serverStore.activeServer, let apiBase = server.url else { return nil }
        return APIClient(apiBase: apiBase, mediaBase: currentMediaBase(for: server), token: token)
    }

    /// The LAN URL when this server has one AND it's proven reachable now;
    /// otherwise nil (APIClient then routes media over the primary URL).
    /// The proven media route's URL, or nil to leave media on the primary URL.
    /// Preference order lives in `ServerConfig.mediaRoutes` (LAN/VPN, then the
    /// direct public backup); this just reads whichever the probe proved.
    private func currentMediaBase(for server: ServerConfig) -> URL? {
        lanReachability.mediaRoute(for: server.id)?.url
    }

    /// LAN reachability flipped: rebuild `api` with the new `mediaBase` if it
    /// actually changed. Publishing `api` makes live views re-resolve their
    /// media URLs on the next attach (a mid-session Wi-Fi→cellular change flips
    /// video back to the HTTPS primary; players re-point on their next play()).
    private func refreshMediaBase() {
        guard let current = api,
              let server = serverStore.activeServer,
              let apiBase = server.url else { return }
        let desired = currentMediaBase(for: server) ?? apiBase
        guard current.mediaBase != desired else { return }
        api = APIClient(apiBase: apiBase, mediaBase: desired, token: current.token)
    }

    // MARK: Private

    private func startSocket() {
        guard let api, let url = api.websocketURL() else { return }
        socket.connect(url: url, protocols: api.wsSubprotocols())
    }

    // Cached identity lets restore() enter the app when the server is
    // temporarily unreachable (offline LAN, VPN down).
    private static func usernameKey(_ id: UUID) -> String { "vigilume.username.\(id.uuidString)" }
    private static func roleKey(_ id: UUID) -> String { "vigilume.role.\(id.uuidString)" }
    /// Pre-rename keys (the app shipped as "Sentinel"), keyed by the same
    /// server uuid — so the migration runs per id, on read.
    private static func legacyUsernameKey(_ id: UUID) -> String { "sentinel.username.\(id.uuidString)" }
    private static func legacyRoleKey(_ id: UUID) -> String { "sentinel.role.\(id.uuidString)" }

    private func cacheIdentity(username: String, role: Role, for id: UUID) {
        UserDefaults.standard.set(username, forKey: Self.usernameKey(id))
        UserDefaults.standard.set(role.rawValue, forKey: Self.roleKey(id))
    }

    /// Read a cached-identity value, adopting the pre-rename key when the new
    /// one is absent. Idempotent (after the first read the legacy key is gone)
    /// and a plain nil when neither exists — losing this only costs the
    /// offline-restore niceties, but there is no reason to drop it.
    private func migratedString(new: String, legacy: String) -> String? {
        let defaults = UserDefaults.standard
        if let value = defaults.string(forKey: new) { return value }
        guard let value = defaults.string(forKey: legacy) else { return nil }
        defaults.set(value, forKey: new)
        defaults.removeObject(forKey: legacy)
        return value
    }

    private func cachedUsername(for id: UUID) -> String? {
        migratedString(new: Self.usernameKey(id), legacy: Self.legacyUsernameKey(id))
    }

    private func cachedRole(for id: UUID) -> Role? {
        migratedString(new: Self.roleKey(id), legacy: Self.legacyRoleKey(id))
            .flatMap(Role.init(rawValue:))
    }
}
