import Combine
import Foundation
import Network

/// App-wide network-path observer for live streaming (docs/ios-design.md §2).
///
/// A single shared `NWPathMonitor` publishes one derived signal, `isConstrained`,
/// that the live layer uses to decide how hard to push video:
///   - grid/list tiles already ride the low-res `_sub` stream, so they never
///     change — but a constrained path keeps the full-screen / single-camera
///     player on SD instead of auto-attempting the HD (often HEVC) main stream;
///   - when the path later becomes unconstrained (Wi-Fi), `LivePlayerModel`
///     auto-upgrades SD → HD (see its upgrade probe).
///
/// `isConstrained` is deliberately conservative: cellular is ALWAYS treated as
/// constrained (interface == .cellular), on top of the OS's own low-data-mode
/// (`path.isConstrained`) and metered/expensive (`path.isExpensive`) flags. So
/// "bad cell quality" never hammers the HD stream, exactly as the owner asked.
@MainActor
final class NetworkQuality: ObservableObject {
    /// Shared instance — one monitor for the whole app.
    static let shared = NetworkQuality()

    /// True when the active path is cellular, expensive/metered, or in the
    /// OS low-data (constrained) mode. Drives the SD-default + auto-upgrade.
    @Published private(set) var isConstrained: Bool

    /// Fires on EVERY path change (even when `isConstrained` is unchanged — e.g.
    /// hopping between two Wi-Fi networks). `LANReachability` subscribes to it to
    /// re-probe the active server's LAN address whenever the network shifts.
    let pathChanged = PassthroughSubject<Void, Never>()

    private let monitor = NWPathMonitor()
    private let queue = DispatchQueue(label: "com.vigilume.networkquality")

    private init() {
        // Seed from the monitor's current path so the first attach isn't a
        // guess (the update handler also fires almost immediately after start).
        isConstrained = Self.derive(from: monitor.currentPath)
        monitor.pathUpdateHandler = { [weak self] path in
            let constrained = Self.derive(from: path)
            Task { @MainActor in
                guard let self else { return }
                // Always announce the path change (LAN re-probe), then update the
                // collapsed constrained flag only when it actually flipped.
                self.pathChanged.send()
                if self.isConstrained != constrained {
                    self.isConstrained = constrained
                }
            }
        }
        monitor.start(queue: queue)
    }

    private nonisolated static func derive(from path: NWPath) -> Bool {
        path.isConstrained
            || path.isExpensive
            || path.usesInterfaceType(.cellular)
    }
}
