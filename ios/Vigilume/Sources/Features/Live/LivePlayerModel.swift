import AVFoundation
import Combine
import Foundation

/// Owns one AVPlayer for a go2rtc live HLS stream (docs/ios-design.md §2).
///
/// go2rtc HLS sessions expire after ~5 s without a segment request, so every
/// resume/retry builds a FRESH AVPlayerItem — a stale item is never reused.
/// Recovery is fully automatic: item failure, playback-stalled notifications,
/// and a progress watchdog all funnel into an exponential-backoff rebuild
/// (1 s → 2 s → 4 s → 8 s → 15 s cap, forever), so a camera or network blip
/// heals without user action.
///
/// **HD → SD fallback** (docs/ios-design.md §2.1.1): `play(primary:fallback:)`
/// takes an optional second URL (the camera substream). If the primary stream
/// fails, or shows no ready frame within ~5 s, the model automatically
/// switches to the fallback (`usingFallback` drives the "SD (compat)" badge;
/// `retryPrimary()` is the "HD" button). If the fallback ALSO fails a few
/// times, `failureText` publishes a real, human-readable reason — the retry
/// loop keeps self-healing in the background regardless.
///
/// **Network-aware default + auto-upgrade** (§2): `play(primary:fallback:preferHD:)`
/// lets the caller start on SD when the path is constrained (bad cellular), so
/// a full-screen view never opens the HD (often HEVC) main on a bad link.
/// Whenever the model is on SD *and* a fallback-capable HD source exists, it
/// silently probes HD in the background (on a network-path improvement, or at
/// most every `upgradeMinInterval` seconds) and only swaps to it once that
/// probe is actually rendering — a seamless SD→HD "auto-correct" with no
/// flapping; if the probe doesn't come up in time it's discarded and SD stays.
@MainActor
final class LivePlayerModel: ObservableObject {
    enum State: Equatable {
        case idle
        case connecting
        case playing
        case retrying(attempt: Int)
    }

    @Published private(set) var state: State = .idle
    @Published private(set) var player: AVPlayer?
    /// True while the SD fallback URL is the active source ("SD (compat)").
    @Published private(set) var usingFallback = false
    /// Non-nil once every configured source has failed: a real error message
    /// for the UI instead of a black box. Cleared when playback recovers.
    @Published private(set) var failureText: String?
    /// Grid tiles stay muted; the single-camera view flips this on tap.
    @Published var isMuted = true {
        didSet { player?.isMuted = isMuted }
    }

    /// The single-camera view keeps the screen awake; grid tiles don't.
    private let keepsScreenAwake: Bool
    /// Grid tiles run the source pair in reverse (low-res sub as primary, the
    /// main stream as a rescue fallback) and must NOT silently probe back to
    /// the "primary" — that would flap a doorbell tile between its broken sub
    /// and the working main. Only the full-screen SD→HD path wants upgrades.
    private let autoUpgrade: Bool

    private var primaryURL: URL?
    private var fallbackURL: URL?
    /// The URL currently attached (primary or fallback).
    private var activeURL: URL?
    /// The active source rendered at least one frame since activation —
    /// a later stall then means "network blip, retry in place", not
    /// "codec/stream is broken, downgrade to SD".
    private var activePlayedOnce = false
    private var attempt = 0
    private var lastProgress = Date()
    private var lastTimeSeconds: Double = .nan
    private var lastItemError: String?
    private var itemCancellables = Set<AnyCancellable>()
    private var retryTask: Task<Void, Never>?
    private var watchdogTask: Task<Void, Never>?
    private var probeTask: Task<Void, Never>?

    // MARK: Auto-upgrade (SD → HD) state
    /// A second, muted AVPlayer that streams the HD main in the background
    /// while SD is on screen. Promoted to `player` only once it renders — so
    /// the swap is seamless. nil unless an upgrade probe is in flight.
    private var upgradePlayer: AVPlayer?
    private var upgradeCancellables = Set<AnyCancellable>()
    private var upgradeTimeoutTask: Task<Void, Never>?
    /// Debounce: never start a new HD probe more often than this.
    private var lastUpgradeAttempt = Date.distantPast
    private var networkCancellable: AnyCancellable?

    /// No frame progress for this long while nominally playing ⇒ rebuild.
    private static let stallWindow: TimeInterval = 12
    private static let watchdogTick: UInt64 = 4_000_000_000  // 4 s
    /// Primary stream gets this long to produce a ready frame before the
    /// model gives up on HD and falls back to SD (when a fallback exists).
    private static let primaryProbeWindow: UInt64 = 5_000_000_000  // 5 s
    /// On the LAST source, this many failed attempts surface `failureText`.
    private static let failureSurfaceAttempts = 2
    /// The background HD probe gets this long to render before it's discarded.
    private static let upgradeProbeWindow: UInt64 = 6_000_000_000  // 6 s
    /// Minimum gap between HD auto-upgrade attempts (anti-flap debounce). A
    /// network-path change bypasses the timer by resetting it; otherwise the
    /// periodic watchdog only re-probes this often.
    private static let upgradeMinInterval: TimeInterval = 25

    init(keepsScreenAwake: Bool = false, autoUpgrade: Bool = true) {
        self.keepsScreenAwake = keepsScreenAwake
        self.autoUpgrade = autoUpgrade
        // A path change to unconstrained is the primary trigger for SD→HD.
        networkCancellable = NetworkQuality.shared.$isConstrained
            .removeDuplicates()
            .receive(on: DispatchQueue.main)
            .sink { [weak self] constrained in
                guard let self, !constrained else { return }
                // New good path: allow an immediate re-probe, then try.
                self.lastUpgradeAttempt = .distantPast
                self.maybeAutoUpgrade()
            }
    }

    deinit {
        retryTask?.cancel()
        watchdogTask?.cancel()
        probeTask?.cancel()
        upgradeTimeoutTask?.cancel()
    }

    // MARK: Lifecycle (tiles call play/stop; scene phase calls suspend/resume)

    /// Attach and start streaming. Safe to call repeatedly with the same URL.
    func play(url: URL) {
        play(primary: url, fallback: nil)
    }

    /// Attach with an optional SD fallback (see class note). HD-first. Safe to
    /// call repeatedly with the same pair.
    func play(primary: URL, fallback: URL?) {
        play(primary: primary, fallback: fallback, preferHD: true)
    }

    /// Attach with an optional SD fallback, choosing the initial quality.
    ///
    /// `preferHD: false` (a constrained/cellular path) starts on the SD
    /// substream so a bad link never opens the HD main — the model still
    /// remembers the HD URL and will auto-upgrade when the path improves.
    /// `preferHD: true` is the classic HD-first behavior. Safe to call
    /// repeatedly with the same arguments.
    func play(primary: URL, fallback: URL?, preferHD: Bool) {
        if primaryURL == primary, fallbackURL == fallback, player != nil,
           state == .playing { return }
        primaryURL = primary
        fallbackURL = fallback
        if let fallback, !preferHD {
            // Start on SD; auto-upgrade takes it to HD when conditions allow.
            activate(url: fallback, asFallback: true)
        } else {
            activate(url: primary, asFallback: false)
        }
    }

    /// Back to the HD stream after a fallback (the "HD" button). Also cancels
    /// any in-flight background upgrade probe — the user asked for HD now.
    func retryPrimary() {
        guard let primaryURL else { return }
        discardUpgrade()
        activate(url: primaryURL, asFallback: false)
    }

    /// Full detach: tear everything down and forget the URLs.
    func stop() {
        primaryURL = nil
        fallbackURL = nil
        activeURL = nil
        cancelRetry()
        stopWatchdog()
        stopProbe()
        discardUpgrade()
        teardownPlayer()
        usingFallback = false
        failureText = nil
        state = .idle
    }

    /// Background the stream but remember the URLs (app went inactive).
    func suspend() {
        cancelRetry()
        stopWatchdog()
        stopProbe()
        discardUpgrade()
        teardownPlayer()
        if activeURL != nil { state = .idle }
    }

    /// Rebuild after suspend() — always a fresh item (see class note).
    func resume() {
        guard activeURL != nil, player == nil else { return }
        attempt = 0
        activePlayedOnce = false
        startWatchdog()
        startPrimaryProbeIfNeeded()
        state = .connecting
        buildPlayer()
    }

    // MARK: Source switching

    /// Point the model at one source (fresh attempt counter + probe).
    private func activate(url: URL, asFallback: Bool) {
        discardUpgrade()
        usingFallback = asFallback
        activeURL = url
        attempt = 0
        activePlayedOnce = false
        failureText = nil
        lastItemError = nil
        cancelRetry()
        startWatchdog()
        startPrimaryProbeIfNeeded()
        state = .connecting
        buildPlayer()
    }

    private func activateFallback() {
        guard let fallbackURL, !usingFallback else { return }
        activate(url: fallbackURL, asFallback: true)
    }

    /// True when a fallback exists and hasn't been switched to yet.
    private var canFallBack: Bool {
        fallbackURL != nil && !usingFallback
    }

    // MARK: Player construction

    private func buildPlayer() {
        guard let url = activeURL else { return }
        teardownPlayer()

        let item = AVPlayerItem(url: url)
        item.preferredForwardBufferDuration = 1  // small buffer = lower latency

        let newPlayer = AVPlayer(playerItem: item)
        newPlayer.automaticallyWaitsToMinimizeStalling = false
        newPlayer.isMuted = isMuted
        newPlayer.preventsDisplaySleepDuringVideoPlayback = keepsScreenAwake

        lastProgress = Date()
        lastTimeSeconds = .nan
        subscribe(to: item, player: newPlayer)
        player = newPlayer
        newPlayer.play()
    }

    private func teardownPlayer() {
        itemCancellables.removeAll()
        player?.pause()
        player?.replaceCurrentItem(with: nil)
        player = nil
    }

    private func subscribe(to item: AVPlayerItem, player: AVPlayer) {
        item.publisher(for: \.status)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] status in
                guard let self, status == .failed else { return }
                self.lastItemError = item.error?.localizedDescription
                self.scheduleRetry()
            }
            .store(in: &itemCancellables)

        player.publisher(for: \.timeControlStatus)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] timeControlStatus in
                guard let self, timeControlStatus == .playing else { return }
                self.attempt = 0
                self.activePlayedOnce = true
                self.lastProgress = Date()
                self.failureText = nil
                self.stopProbe()
                if self.state != .playing { self.state = .playing }
            }
            .store(in: &itemCancellables)

        let failureNotifications: [Notification.Name] = [
            .AVPlayerItemPlaybackStalled,
            .AVPlayerItemFailedToPlayToEndTime,
        ]
        for name in failureNotifications {
            NotificationCenter.default.publisher(for: name, object: item)
                .receive(on: DispatchQueue.main)
                .sink { [weak self] _ in self?.scheduleRetry() }
                .store(in: &itemCancellables)
        }
    }

    // MARK: Primary probe (no ready frame in ~5 s ⇒ fall back to SD)

    private func startPrimaryProbeIfNeeded() {
        stopProbe()
        guard canFallBack else { return }
        probeTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: Self.primaryProbeWindow)
            if Task.isCancelled { return }
            guard let self, self.state != .playing, self.canFallBack else { return }
            if self.lastItemError == nil {
                self.lastItemError = "Timed out waiting for the HD stream."
            }
            self.activateFallback()
        }
    }

    private func stopProbe() {
        probeTask?.cancel()
        probeTask = nil
    }

    // MARK: Stall watchdog

    private func startWatchdog() {
        stopWatchdog()
        watchdogTask = Task { [weak self] in
            while true {
                try? await Task.sleep(nanoseconds: Self.watchdogTick)
                if Task.isCancelled { return }
                guard let self else { return }
                self.watchdogCheck()
            }
        }
    }

    private func stopWatchdog() {
        watchdogTask?.cancel()
        watchdogTask = nil
    }

    private func watchdogCheck() {
        // Periodic (debounced) SD→HD re-probe when we're stuck on SD but the
        // path is good — covers the "conditions improved but no path-change
        // event fired" case without a tight loop.
        maybeAutoUpgrade()

        guard let player, state == .playing || state == .connecting else { return }
        let seconds = player.currentTime().seconds
        if seconds.isFinite, seconds != lastTimeSeconds {
            lastTimeSeconds = seconds
            lastProgress = Date()
        } else if Date().timeIntervalSince(lastProgress) > Self.stallWindow {
            scheduleRetry()
        }
    }

    // MARK: Auto-upgrade (SD → HD, seamless)

    /// True when an HD source exists that we aren't currently showing — i.e.
    /// a distinct main stream is configured and SD is the active source.
    private var canUpgradeToHD: Bool {
        guard usingFallback, let primaryURL, let fallbackURL else { return false }
        return primaryURL != fallbackURL
    }

    /// Start a background HD probe if we're on SD, the path is good, no probe
    /// is already running, and the debounce window has elapsed.
    private func maybeAutoUpgrade() {
        guard autoUpgrade else { return }
        guard canUpgradeToHD, let primaryURL else { return }
        guard upgradePlayer == nil else { return }
        guard !NetworkQuality.shared.isConstrained else { return }
        guard Date().timeIntervalSince(lastUpgradeAttempt) >= Self.upgradeMinInterval else { return }
        lastUpgradeAttempt = Date()
        startUpgradeProbe(to: primaryURL)
    }

    /// Build a silent, off-screen HD player. If it reaches `.playing` within
    /// the window it's promoted to the visible player (SD dropped); otherwise
    /// it's discarded and SD stays — no flicker, no black frame.
    private func startUpgradeProbe(to url: URL) {
        discardUpgrade()
        let item = AVPlayerItem(url: url)
        item.preferredForwardBufferDuration = 1

        let probe = AVPlayer(playerItem: item)
        probe.automaticallyWaitsToMinimizeStalling = false
        probe.isMuted = true  // stays silent until promoted
        upgradePlayer = probe

        probe.publisher(for: \.timeControlStatus)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] status in
                guard let self, status == .playing else { return }
                self.promoteUpgrade()
            }
            .store(in: &upgradeCancellables)

        probe.play()

        upgradeTimeoutTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: Self.upgradeProbeWindow)
            if Task.isCancelled { return }
            guard let self else { return }
            // Didn't render in time — throw it away, stay on SD.
            if self.upgradePlayer != nil { self.discardUpgrade() }
        }
    }

    /// The HD probe is rendering: make it the live player and drop SD.
    private func promoteUpgrade() {
        guard let probe = upgradePlayer, let primaryURL else { return }
        upgradeCancellables.removeAll()
        upgradeTimeoutTask?.cancel()
        upgradeTimeoutTask = nil
        upgradePlayer = nil

        // Swap the SD player out for the already-playing HD one.
        teardownPlayer()
        activeURL = primaryURL
        usingFallback = false
        attempt = 0
        activePlayedOnce = true
        failureText = nil
        lastItemError = nil
        lastProgress = Date()
        lastTimeSeconds = .nan

        probe.isMuted = isMuted
        probe.preventsDisplaySleepDuringVideoPlayback = keepsScreenAwake
        if let item = probe.currentItem {
            subscribe(to: item, player: probe)
        }
        player = probe
        state = .playing
        startWatchdog()
    }

    private func discardUpgrade() {
        upgradeCancellables.removeAll()
        upgradeTimeoutTask?.cancel()
        upgradeTimeoutTask = nil
        upgradePlayer?.pause()
        upgradePlayer?.replaceCurrentItem(with: nil)
        upgradePlayer = nil
    }

    // MARK: Retry

    private func scheduleRetry() {
        guard activeURL != nil else { return }

        // Failure on the HD stream BEFORE it ever rendered, with an SD
        // fallback configured: switch immediately instead of backing off on
        // a source that likely can't play at all (e.g. codec rejection).
        if canFallBack, !activePlayedOnce {
            activateFallback()
            return
        }

        guard retryTask == nil else { return }
        teardownPlayer()
        attempt += 1
        state = .retrying(attempt: attempt)

        // Last (or only) source keeps failing: surface a real reason while
        // the backoff loop continues to self-heal underneath.
        if attempt >= Self.failureSurfaceAttempts {
            failureText = lastItemError ?? "The camera stream isn't responding."
        }

        let delay = min(15.0, Double(1 << min(attempt - 1, 4)))
        retryTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
            guard let self else { return }
            if Task.isCancelled { return }
            self.retryTask = nil
            guard self.activeURL != nil else { return }
            self.state = .connecting
            self.buildPlayer()
        }
    }

    private func cancelRetry() {
        retryTask?.cancel()
        retryTask = nil
    }
}
