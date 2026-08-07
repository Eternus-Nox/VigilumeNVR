import Combine
import Foundation
import UIKit

/// Which path media (video) is currently taking for a server.
///
/// Control/auth/list traffic ALWAYS rides the primary URL. Only media is
/// routed, because only media is big enough for the pipe to matter.
enum MediaRoute: Equatable, Sendable {
    /// The LAN or VPN address — the only routed path. Fast, direct, encrypted
    /// end to end over a VPN, and needs nothing exposed to the internet. When
    /// it is not reachable, media falls back to the primary URL (represented
    /// by the absence of a route, not by a case here).
    case lan(URL)

    var url: URL {
        switch self {
        case .lan(let u): return u
        }
    }

    var isLAN: Bool {
        if case .lan = self { return true }
        return false
    }
}

/// Per-server MEDIA ROUTE probe.
///
/// A saved server may carry, besides its primary (usually HTTPS/CDN) URL:
///   - a LAN/VPN address — preferred for video whenever it answers, and
///   - a direct public address — the backup when it does not.
///
/// WHY THIS EXISTS AT ALL. The primary URL is typically fronted by a CDN
/// tunnel, which carries the control API perfectly but is a poor pipe for
/// sustained video: segment fetches get buffered/throttled (and most CDN terms
/// restrict serving video outright). That is what makes remote live view
/// stutter while everything else in the app feels fine. Routing ONLY media off
/// that pipe fixes live view without giving up the CDN for anything else.
///
/// **How it decides.** For each candidate it fires a fast unauthenticated
/// `GET <candidate>/api/system/health` with a short (~1.5 s) timeout.
/// "Reachable" means the host *answered at all* — any HTTP response (even
/// 401/404) proves it is routable — so a probe failure is strictly a transport
/// failure/timeout. Candidates are probed CONCURRENTLY but the winner is chosen
/// by the server's declared PREFERENCE ORDER, so a slightly slower LAN/VPN
/// still beats a faster public path. The probe never blocks the UI, and the
/// value defaults to **no route until proven**, so media stays on the primary
/// URL unless something better answers.
///
/// **When it re-probes.** On `setActiveServer` (login / server switch), on every
/// network path change (`NetworkQuality.pathChanged` — covers leaving Wi-Fi for
/// cellular, or hopping Wi-Fi networks, or a VPN coming up/down), and on app
/// foreground. `SessionModel` observes `routes` and rebuilds its `APIClient`
/// with a new `mediaBase` whenever the active server's answer changes; live
/// players re-resolve their URLs on the next attach.
@MainActor
final class LANReachability: ObservableObject {
    /// One probe for the whole app.
    static let shared = LANReachability()

    /// Per-server resolved media route. Absent == none proven == use the
    /// primary URL.
    @Published private(set) var routes: [UUID: MediaRoute] = [:]

    private var activeServer: ServerConfig?
    private var probeTask: Task<Void, Never>?
    private var cancellables = Set<AnyCancellable>()

    /// Fast, connectivity-strict session so a dead host times out quickly
    /// instead of waiting on the OS default (~60 s).
    private static let probeSession: URLSession = {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 1.5
        config.timeoutIntervalForResource = 1.5
        config.waitsForConnectivity = false
        return URLSession(configuration: config)
    }()

    private init() {
        // Re-probe whenever the network path shifts (Wi-Fi ⇄ cellular, a
        // different Wi-Fi, or a VPN toggling). A LAN address is meaningless off
        // its own network, and a VPN coming up makes one meaningful again.
        NetworkQuality.shared.pathChanged
            .sink { [weak self] in self?.reprobeActive() }
            .store(in: &cancellables)

        // Re-probe on foreground: the device may have joined/left the LAN (or
        // the VPN) while the app was backgrounded.
        NotificationCenter.default.publisher(for: UIApplication.didBecomeActiveNotification)
            .sink { [weak self] _ in self?.reprobeActive() }
            .store(in: &cancellables)
    }

    // MARK: Query

    /// The proven media route for a server, or nil to use the primary URL.
    func mediaRoute(for id: UUID) -> MediaRoute? {
        routes[id]
    }

    /// True only when the server's LAN/VPN address is the proven route. Drives
    /// the "on LAN" badge in Settings.
    func lanReachable(for id: UUID) -> Bool {
        routes[id]?.isLAN ?? false
    }

    // MARK: Active server

    /// Point the probe at the active server (login / server switch). Clears any
    /// stale answer for it — we default to no-route until this probe proves
    /// otherwise — then verifies now.
    func setActiveServer(_ server: ServerConfig?) {
        activeServer = server
        probeTask?.cancel()
        guard let server else { return }
        // Fresh server selection: don't inherit a prior answer; prove it again.
        setRoute(nil, for: server.id)
        guard !server.mediaRoutes().isEmpty else { return }
        startProbe(server)
    }

    /// Re-verify the current active server without clearing its last-known
    /// answer first (avoids flapping the media base on a transient same-network
    /// blip). A genuine loss simply fails the probe and clears the route.
    private func reprobeActive() {
        guard let server = activeServer else { return }
        guard !server.mediaRoutes().isEmpty else {
            setRoute(nil, for: server.id)
            return
        }
        startProbe(server)
    }

    /// Probe every candidate CONCURRENTLY, then pick by PREFERENCE ORDER.
    ///
    /// Sequential probing would cost a full timeout per dead candidate before
    /// reaching a live one — seconds of no video every time you are off the
    /// VPN. Choosing by index rather than by who answered first keeps the
    /// preference honest: a slightly slower LAN/VPN still beats a faster
    /// public path.
    private func startProbe(_ server: ServerConfig) {
        probeTask?.cancel()
        let id = server.id
        let candidates = server.mediaRoutes()
        guard !candidates.isEmpty else {
            setRoute(nil, for: id)
            return
        }
        probeTask = Task { [weak self] in
            guard let self else { return }
            var answered = [Bool](repeating: false, count: candidates.count)
            await withTaskGroup(of: (Int, Bool).self) { group in
                for (index, candidate) in candidates.enumerated() {
                    group.addTask { (index, await Self.answers(candidate.url)) }
                }
                for await (index, ok) in group { answered[index] = ok }
            }
            guard !Task.isCancelled, self.activeServer?.id == id else { return }
            self.setRoute(candidates.enumerated().first { answered[$0.offset] }?.element,
                          for: id)
        }
    }

    /// True when the host answers at all — any HTTP status proves it is
    /// routable from here, which is the only question being asked.
    private static func answers(_ base: URL) async -> Bool {
        var request = URLRequest(url: base.appendingPathComponent("api/system/health"))
        request.httpMethod = "GET"
        request.timeoutInterval = 1.5
        request.cachePolicy = .reloadIgnoringLocalCacheData
        do {
            let (_, response) = try await probeSession.data(for: request)
            return (response as? HTTPURLResponse) != nil
        } catch {
            return false
        }
    }

    private func setRoute(_ value: MediaRoute?, for id: UUID) {
        if routes[id] != value { routes[id] = value }
    }
}
