import AVFoundation
import Combine
import Foundation
import SwiftUI
import WebRTC

/// The live-view brain: **WHEP-first with automatic HLS fallback**
/// (docs/ios-design.md §2). Every live surface (grid tile, full-screen,
/// camera-detail card) owns one of these instead of a bare `LivePlayerModel`.
///
/// It composes both players and picks between them:
///   1. On `play`, start a `WHEPPlayer` (sub-second WebRTC) and a fallback
///      timer.
///   2. If WHEP renders a real frame within ~4.5 s, stay on WebRTC.
///   3. If WHEP errors, or the timer fires before a frame, fall back to the
///      **unchanged** `LivePlayerModel` HLS path — which keeps ALL of its own
///      behavior (HD↔SD substream fallback, network-aware default, seamless
///      SD→HD auto-upgrade, watchdog/backoff self-heal, real error text). So
///      live never breaks on WebRTC-hostile networks.
///
/// The public surface deliberately mirrors `LivePlayerModel` (`state`,
/// `usingFallback`, `failureText`, `isMuted`, `stop`/`suspend`/`resume`/
/// `retryPrimary`) so the three call sites change as little as possible; the
/// only new argument is the `whep:` URL. `state` reuses `LivePlayerModel.State`
/// so `PlayerStateOverlay` is untouched.
@MainActor
final class LiveController: ObservableObject {
    enum Renderer: Equatable { case none, whep, hls }

    @Published private(set) var state: LivePlayerModel.State = .idle
    /// True while the HLS SD substream is the active fallback source (drives the
    /// "SD (compat)" badge). Always false on the WebRTC path.
    @Published private(set) var usingFallback = false
    /// Real, human-readable reason when every source failed (from the HLS model).
    @Published private(set) var failureText: String?

    /// Which renderer the view should show right now.
    @Published private(set) var mode: Renderer = .none
    /// The WebRTC video track (valid when `mode == .whep`).
    @Published private(set) var whepTrack: RTCVideoTrack?
    /// The HLS AVPlayer (valid when `mode == .hls`).
    @Published private(set) var hlsPlayer: AVPlayer?

    /// Grid tiles stay muted; the single-camera views flip this on tap. Applies
    /// to whichever player is active.
    @Published var isMuted = true {
        didSet {
            whep.isMuted = isMuted
            standby?.isMuted = isMuted
            hls.isMuted = isMuted
            // AUDIO LIVES ONLY ON THE HIGH RUNG. go2rtc attaches the
            // `#audio=aac` source to `<cam>` only; `<cam>_sub` is raw substream
            // video and most of these cameras don't even encode substream audio.
            // So "unmute" means "I want sound", which means we must be on main.
            // Pinning here also makes the whole switcher safe: it only ever runs
            // while muted, so two live players can never both be fighting over
            // the reference-counted audio session.
            if !isMuted, adaptEnabled, rung == .low, mode == .whep {
                switchTo(.high, force: true)
            }
        }
    }

    /// The player currently ON SCREEN. A `var` because a quality switch swaps in
    /// the standby player wholesale (make-before-break, see `switchTo`).
    private var whep: WHEPPlayer
    /// Candidate rung being brought up in parallel during a switch. It renders
    /// nowhere until it has produced a real frame; only then does it become
    /// `whep`. This is what keeps a switch from blanking the view.
    private var standby: WHEPPlayer?
    private let hls: LivePlayerModel
    /// Remembered so a standby player is built exactly like the active one (a
    /// grid tile must never negotiate audio — see the `allowsAudio` note above).
    private let allowsAudio: Bool

    private var cancellables = Set<AnyCancellable>()
    /// Sinks bound to the ACTIVE WHEP player only. Cleared + re-bound on swap,
    /// so the retired player can never publish into the view.
    private var whepBindings = Set<AnyCancellable>()
    private var fallbackTask: Task<Void, Never>?
    private var switchTask: Task<Void, Never>?

    // Stored attach parameters (for fallback/resume/retry).
    private var whepURL: URL?
    private var primaryURL: URL?
    private var fallbackURL: URL?
    private var preferHD = true

    // MARK: Adaptive quality (rung switching)

    /// Which go2rtc source is on screen. There is no bitrate adaptation to be
    /// had — go2rtc repacks the camera's fixed encode — so "quality" is purely
    /// WHICH STREAM we ask for: `<cam>_sub` or `<cam>`.
    enum Rung: Equatable { case low, high }

    @Published private(set) var rung: Rung = .low
    /// True once a promotion to the full-res rung has been TRIED and failed —
    /// the candidate never decoded a frame inside `candidateWindow`. Drives the
    /// "SD (compat)" badge on the WebRTC path, so a view that is stuck on the
    /// substream says so instead of just looking soft. Cleared by a successful
    /// switch, a fresh attach, and the HD retry button.
    @Published private(set) var highRungUnavailable = false
    private var whepLowURL: URL?
    private var whepHighURL: URL?
    /// False when there is nothing to switch between: no high rung (grid tiles,
    /// which must stay small), or high == low (the AD410 doorbell, whose `_sub`
    /// resolves to the same source as main — pulling a second session there is
    /// what broke its talk backchannel once).
    private var adaptEnabled = false

    private var lastSwitchAt: TimeInterval = 0
    private var degradedSince: TimeInterval?
    private var cleanSince: TimeInterval?
    private var lastSample: LiveQualitySample?
    /// Clean time required before climbing to the high rung. Starts SHORT (see
    /// `initialPromoteWindow`) and only becomes cautious once this link has
    /// actually failed at high.
    private var promoteWindow: TimeInterval = LiveController.initialPromoteWindow
    /// Has the high rung ever failed on THIS attach? Until it has, there is no
    /// evidence to be cautious about, so the first climb is fast and unthrottled.
    private var hasDemoted = false

    /// Demote after this long degraded — short, because the picture is already
    /// bad and waiting helps nobody.
    private static let demoteWindow: TimeInterval = 10
    /// FIRST climb off the opening rung: ~2 clean samples. Opening low exists to
    /// put a frame on screen fast, NOT to make you watch a soft image — on a
    /// healthy link full res should arrive within a few seconds. The swap is
    /// make-before-break, so this costs a brief second connection and no visible
    /// interruption. (20 s here was the wrong call: it left a deliberate
    /// fullscreen open looking soft for far too long on a perfectly good LAN.)
    private static let initialPromoteWindow: TimeInterval = 4
    /// Clean time required to climb AFTER a demote — this is where anti-flap
    /// actually matters, because the link has now proven it can't always hold
    /// high. Doubles on each further demote.
    private static let cautiousPromoteWindow: TimeInterval = 20
    /// Never switch more often than this — but only ONCE the link has misbehaved
    /// (`hasDemoted`); it must not throttle the first climb.
    private static let minSwitchInterval: TimeInterval = 25
    /// A candidate rung gets this long to produce a real frame before we give up
    /// and stay where we are.
    private static let candidateWindow: TimeInterval = 6
    private static let promoteWindowMax: TimeInterval = 120

    /// Dedupe key so repeated `play(...)` calls (SwiftUI onChange storms) don't
    /// restart a live session.
    private struct Key: Equatable {
        let whep: URL?
        let whepHigh: URL?
        let primary: URL
        let fallback: URL?
    }
    private var currentKey: Key?
    /// True only between a `suspend()` and the next `resume()`/`play()`.
    private var suspended = false

    /// WHEP gets this long to produce a real frame before we fall back to HLS.
    private static let whepFallbackWindow: UInt64 = 4_500_000_000  // 4.5 s

    /// - Parameter allowsAudio: pass `false` for permanently-muted surfaces (the
    ///   grid tiles). It stops WHEP negotiating an audio track at all, which is
    ///   what keeps WebRTC from opening the audio unit — and therefore the MIC —
    ///   for tiles nobody is listening to.
    init(keepsScreenAwake: Bool = false, autoUpgrade: Bool = true, allowsAudio: Bool = true) {
        self.allowsAudio = allowsAudio
        whep = WHEPPlayer(allowsAudio: allowsAudio)
        hls = LivePlayerModel(keepsScreenAwake: keepsScreenAwake, autoUpgrade: autoUpgrade)
        bindChildren()
    }

    deinit {
        fallbackTask?.cancel()
    }

    // MARK: Child observation

    private func bindChildren() {
        bindActiveWHEP()

        // HLS: republish the model's own state while it's the active renderer.
        hls.$state
            .receive(on: DispatchQueue.main)
            .sink { [weak self] s in
                guard let self, self.mode == .hls else { return }
                self.state = s
            }
            .store(in: &cancellables)

        hls.$usingFallback
            .receive(on: DispatchQueue.main)
            .sink { [weak self] v in
                guard let self, self.mode == .hls else { return }
                self.usingFallback = v
            }
            .store(in: &cancellables)

        hls.$failureText
            .receive(on: DispatchQueue.main)
            .sink { [weak self] t in
                guard let self, self.mode == .hls else { return }
                self.failureText = t
            }
            .store(in: &cancellables)

        hls.$player
            .receive(on: DispatchQueue.main)
            .sink { [weak self] p in
                guard let self, self.mode == .hls else { return }
                self.hlsPlayer = p
            }
            .store(in: &cancellables)
    }

    /// (Re)bind the view-facing publishers to whichever player is active.
    /// Clearing `whepBindings` first is what makes a swap safe: the retired
    /// player's sinks are torn down before the new one is wired up, so a
    /// late frame from the old session can never repaint the view.
    private func bindActiveWHEP() {
        whepBindings.removeAll()

        whep.$videoTrack
            .receive(on: DispatchQueue.main)
            .sink { [weak self] track in
                guard let self, self.mode == .whep else { return }
                self.whepTrack = track
            }
            .store(in: &whepBindings)

        whep.$state
            .receive(on: DispatchQueue.main)
            .sink { [weak self] whepState in
                guard let self, self.mode == .whep else { return }
                self.handleWHEPState(whepState)
            }
            .store(in: &whepBindings)

        // Link telemetry — the only input the rung switcher has.
        whep.$quality
            .receive(on: DispatchQueue.main)
            .sink { [weak self] sample in
                guard let self, let sample else { return }
                self.evaluateQuality(sample)
            }
            .store(in: &whepBindings)
    }

    // MARK: Public lifecycle (mirrors LivePlayerModel + a `whep:` URL)

    /// Attach WHEP-first with an HLS fallback pair. `whep` is the WebRTC
    /// endpoint (nil ⇒ skip straight to HLS); `primary`/`fallback` are the
    /// existing HLS URLs handed to `LivePlayerModel` verbatim.
    /// - Parameters:
    ///   - whepLow: the SMALL rung (`<cam>_sub`). Playback always starts here
    ///     when present: it connects sooner, decodes sooner, and survives a weak
    ///     link — "load fast" in practice.
    ///   - whepHigh: the full-res rung (`<cam>`), used as the promotion target
    ///     once the link proves itself. Pass `nil` to pin LOW forever (grid
    ///     tiles: a wall of full-res sessions is exactly what we're avoiding).
    func play(
        whepLow: URL?,
        whepHigh: URL? = nil,
        primary: URL,
        fallback: URL?,
        preferHD: Bool = true
    ) {
        let key = Key(whep: whepLow, whepHigh: whepHigh, primary: primary, fallback: fallback)
        // Already running this exact source (and not parked by suspend): no-op.
        if key == currentKey, !suspended, mode != .none { return }

        currentKey = key
        self.whepLowURL = whepLow
        self.whepHighURL = whepHigh
        // Adapt only when there are genuinely two different rungs to move
        // between. Equal URLs means the camera has no real substream (the AD410
        // doorbell's `_sub` resolves to its main source) — switching there would
        // buy nothing and would open a second RTSP session against a camera
        // whose talk backchannel needs a free one.
        self.adaptEnabled = whepHigh != nil && whepLow != nil && whepHigh != whepLow
        // Start LOW when we have it. `preferHD` no longer decides the opening
        // rung — the link's measured behaviour does.
        self.rung = whepLow != nil ? .low : .high
        self.whepURL = whepLow ?? whepHigh
        self.primaryURL = primary
        self.fallbackURL = fallback
        self.preferHD = preferHD
        resetQualityTracking()
        startWHEPAttempt()
    }

    /// Clear the rolling quality state — on a fresh attach nothing that came
    /// before is evidence about this link.
    private func resetQualityTracking() {
        cancelSwitch()
        lastSample = nil
        degradedSince = nil
        cleanSince = nil
        promoteWindow = Self.initialPromoteWindow
        hasDemoted = false
        highRungUnavailable = false
        lastSwitchAt = ProcessInfo.processInfo.systemUptime
    }

    private func url(for rung: Rung) -> URL? {
        rung == .low ? whepLowURL : whepHighURL
    }

    // MARK: Adaptive rung switching

    /// Fold one stats sample into the demote/promote decision.
    ///
    /// The asymmetry is the whole design: DROP FAST (the picture is already
    /// breaking up — waiting only prolongs it) and CLIMB SLOWLY (a few good
    /// seconds prove nothing, and a wrong promotion is visible). Combined with a
    /// hard floor between switches and a growing promote window after each
    /// demote, a marginal link settles on the low rung instead of oscillating.
    private func evaluateQuality(_ sample: LiveQualitySample) {
        defer { lastSample = sample }
        guard adaptEnabled, mode == .whep, state == .playing, standby == nil else { return }
        guard let previous = lastSample, sample.at > previous.at else { return }

        let now = sample.at
        let elapsed = sample.at - previous.at
        let freezeDelta = max(0, sample.freezeCount - previous.freezeCount)
        let lossDelta = max(0, sample.packetsLost - previous.packetsLost)
        let decodedDelta = max(0, sample.framesDecoded - previous.framesDecoded)
        // Prefer WebRTC's own fps; fall back to the decoded-frame delta so a
        // build that omits `framesPerSecond` still gets a real reading.
        let fps = sample.framesPerSecond > 0
            ? sample.framesPerSecond
            : Double(decodedDelta) / max(elapsed, 0.001)

        // Any ONE of these means the link is not carrying this rung: the stream
        // visibly stalled, packets are being lost in bulk, or decode has
        // collapsed. The 3 fps floor is deliberately low — several of these
        // cameras run 5-6 fps substreams legitimately.
        let degraded = freezeDelta >= 2 || lossDelta >= 20 || fps < 3

        if degraded {
            cleanSince = nil
            let since = degradedSince ?? now
            degradedSince = since
            // DEMOTE only while muted. Unmuted we are pinned to the
            // audio-bearing rung — `<cam>_sub` carries no audio at all, so
            // dropping would silence the stream the user unmuted on purpose.
            // Promotion, below, runs either way: it is the direction that GIVES
            // you audio, and it is what a fullscreen view needs most.
            guard rung == .high, isMuted else { return }
            if now - since >= Self.demoteWindow, now - lastSwitchAt >= Self.minSwitchInterval {
                // This link has now failed at high, so future climbs must earn
                // it: jump to the cautious window on the first failure, then
                // double on each one after that.
                promoteWindow = hasDemoted
                    ? min(promoteWindow * 2, Self.promoteWindowMax)
                    : Self.cautiousPromoteWindow
                hasDemoted = true
                switchTo(.low)
            }
        } else {
            degradedSince = nil
            let since = cleanSince ?? now
            cleanSince = since
            // THE RETRY PATH. `handleWHEPState` fires one make-before-break
            // promotion when the sub rung first paints; if that attempt loses
            // the race (main's keyframe interval is untouched by
            // provision_substream_gop, so its first I-frame can arrive after
            // the 6 s candidate window) this is the only thing that tries
            // again. It used to be unreachable on the single-camera fullscreen
            // view, which is unmuted from the moment it opens — so one unlucky
            // promotion left that view on the 640x480 substream, full-screen,
            // for as long as it stayed open.
            guard rung == .low, whepHighURL != nil else { return }
            // The rate limiter exists to stop FLAPPING, which can only happen
            // once something has flapped. Before the first demote there is
            // nothing to damp, so the opening climb isn't throttled by it.
            let throttled = hasDemoted && (now - lastSwitchAt < Self.minSwitchInterval)
            if now - since >= promoteWindow, !throttled {
                switchTo(.high, force: !hasDemoted)
            }
        }
    }

    /// Bring the other rung up ALONGSIDE the current one and only swap when the
    /// candidate has painted a real frame (`WHEPPlayer` reports `.playing` on a
    /// decoded frame, not on ICE). If it never does, we stay put — a failed
    /// switch must never cost the user the stream they already had.
    ///
    /// - Parameter force: skip the rate limit (used by unmute, which is a direct
    ///   user action rather than an inference about the network).
    private func switchTo(_ target: Rung, force: Bool = false) {
        guard mode == .whep, target != rung, standby == nil, let url = url(for: target) else { return }
        if !force, ProcessInfo.processInfo.systemUptime - lastSwitchAt < Self.minSwitchInterval { return }

        let candidate = WHEPPlayer(allowsAudio: allowsAudio)
        // ALWAYS muted, even when the view is unmuted. A candidate renders
        // nowhere until it commits, but an unmuted one would enable its own
        // remote audio track AND activate the shared RTCAudioSession while the
        // player on screen still holds both — two sources into one VPIO unit
        // for the length of the overlap. `commitSwitch` applies the real mute
        // state at the instant this player becomes the visible one.
        candidate.isMuted = true
        standby = candidate
        candidate.start(url: url)

        switchTask = Task { [weak self] in
            let deadline = ProcessInfo.processInfo.systemUptime + Self.candidateWindow
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 200_000_000)
                if Task.isCancelled { return }
                guard let self, self.standby === candidate else { return }
                if candidate.state == .playing {
                    self.commitSwitch(to: target, candidate: candidate)
                    return
                }
                if candidate.state == .failed
                    || ProcessInfo.processInfo.systemUptime > deadline {
                    self.abandonSwitch(candidate, target: target)
                    return
                }
            }
        }
    }

    /// The candidate is rendering: promote it to the visible player and retire
    /// the old one. The view never sees a gap — `whepTrack` goes straight from
    /// the old track to an already-decoding one.
    private func commitSwitch(to target: Rung, candidate: WHEPPlayer) {
        guard standby === candidate else { return }
        let retired = whep
        whep = candidate
        standby = nil
        switchTask = nil
        bindActiveWHEP()
        whep.isMuted = isMuted
        whepTrack = candidate.videoTrack
        state = .playing
        rung = target
        if target == .high { highRungUnavailable = false }
        whepURL = url(for: target)
        lastSwitchAt = ProcessInfo.processInfo.systemUptime
        degradedSince = nil
        cleanSince = nil
        lastSample = nil
        retired.stop()
    }

    /// Candidate never produced a frame (or failed): drop it and stay put.
    private func abandonSwitch(_ candidate: WHEPPlayer, target: Rung) {
        guard standby === candidate else { return }
        standby = nil
        switchTask = nil
        candidate.stop()
        // Treat a failed attempt as evidence too, so we don't retry immediately.
        lastSwitchAt = ProcessInfo.processInfo.systemUptime
        cleanSince = nil
        // A failed CLIMB has to cost what a demote costs, not less. A high rung
        // that cannot come up at all — a main stream in HEVC, which no WebRTC
        // decoder on this platform can carry, or a camera with no RTSP session
        // slot left — fails this way every single time. Leaving `promoteWindow`
        // at its opening 4 s meant re-dialling main every few seconds for as
        // long as the view stayed open: a new WHEP session, a new RTSP pull off
        // the camera, forever, all of it invisible. Charging the cautious
        // window (then doubling, and switching the rate limiter on) makes a
        // hopeless rung back off on its own while a merely unlucky one still
        // recovers. A failed DEMOTE is not evidence about climbing, so it is
        // charged nothing beyond the rate limit above.
        guard target == .high else { return }
        promoteWindow = hasDemoted
            ? min(promoteWindow * 2, Self.promoteWindowMax)
            : Self.cautiousPromoteWindow
        hasDemoted = true
        highRungUnavailable = true
    }

    private func cancelSwitch() {
        switchTask?.cancel()
        switchTask = nil
        standby?.stop()
        standby = nil
    }

    /// Full detach: forget the source and tear both players down.
    func stop() {
        currentKey = nil
        whepURL = nil
        whepLowURL = nil
        whepHighURL = nil
        adaptEnabled = false
        highRungUnavailable = false
        primaryURL = nil
        fallbackURL = nil
        suspended = false
        cancelFallbackTimer()
        cancelSwitch()
        whep.stop()
        hls.stop()
        whepTrack = nil
        hlsPlayer = nil
        mode = .none
        usingFallback = false
        failureText = nil
        state = .idle
    }

    /// App went inactive: drop both players (WebRTC can't run backgrounded) but
    /// remember the source for `resume()`.
    func suspend() {
        cancelFallbackTimer()
        cancelSwitch()
        whep.stop()
        hls.suspend()
        whepTrack = nil
        hlsPlayer = nil
        mode = .none
        suspended = true
        if primaryURL != nil { state = .idle }
    }

    /// Rebuild after `suspend()` — WHEP-first again for a low-latency resume.
    func resume() {
        guard suspended, primaryURL != nil else { return }
        startWHEPAttempt()
    }

    /// The "HD" button after an SD fallback: re-attempt the low-latency WHEP
    /// main; if it can't come up, HLS falls back to HD (preferHD forced true).
    func retryPrimary() {
        preferHD = true
        if whepURL != nil {
            // An explicit "give me HD" — jump to the high rung if there is one
            // and let the measured link demote it again if it can't hold.
            if let high = whepHighURL {
                rung = .high
                whepURL = high
            }
            resetQualityTracking()
            startWHEPAttempt()
        } else {
            mode = .hls
            hls.retryPrimary()
        }
    }

    // MARK: WHEP attempt + fallback

    private func startWHEPAttempt() {
        suspended = false
        cancelFallbackTimer()
        // Never keep two live sessions alive at once.
        hls.stop()
        hlsPlayer = nil
        usingFallback = false
        failureText = nil

        guard let whepURL else {
            fallBackToHLS()
            return
        }

        mode = .whep
        state = .connecting
        whepTrack = nil
        whep.isMuted = isMuted
        whep.start(url: whepURL)

        fallbackTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: Self.whepFallbackWindow)
            if Task.isCancelled { return }
            guard let self, self.mode == .whep, self.state != .playing else { return }
            // No decodable frame in time — hand off to HLS.
            self.fallBackToHLS()
        }
    }

    private func handleWHEPState(_ whepState: WHEPPlayer.State) {
        switch whepState {
        case .connecting:
            if state != .playing { state = .connecting }
        case .playing:
            cancelFallbackTimer()
            whepTrack = whep.videoTrack
            usingFallback = false
            failureText = nil
            state = .playing
            // The low rung is on screen — now go straight for full res. We do
            // NOT wait for a clean-stats window here: the small stream exists to
            // put a frame up FAST, not to be what you end up watching. Waiting
            // meant a single-camera view sat visibly soft for seconds on a link
            // that was obviously fine.
            //
            // Safe because the swap is make-before-break: the high rung is built
            // alongside this one and only takes over once IT has decoded a
            // frame, so a link that can't carry it simply never swaps (and the
            // stats path then demotes + turns cautious). Skipped once this link
            // has already failed at high — that's when evidence beats optimism.
            if adaptEnabled, rung == .low, !hasDemoted, standby == nil {
                switchTo(.high, force: true)
            }
        case .failed:
            fallBackToHLS()
        case .idle:
            break
        }
    }

    private func fallBackToHLS() {
        cancelFallbackTimer()
        whep.stop()
        whepTrack = nil
        guard let primaryURL else {
            mode = .none
            state = .idle
            return
        }
        mode = .hls
        hls.isMuted = isMuted
        // Hand the HLS model its untouched HD↔SD pair; it takes over state,
        // usingFallback, failureText and player via the bound sinks.
        hls.play(primary: primaryURL, fallback: fallbackURL, preferHD: preferHD)
    }

    private func cancelFallbackTimer() {
        fallbackTask?.cancel()
        fallbackTask = nil
    }
}

/// Renders whichever player `LiveController` has chosen — the WebRTC Metal view
/// or the HLS `AVPlayerLayer` — at the given gravity. Shows nothing while idle
/// (the caller's `PlayerStateOverlay` draws the spinner/placeholder on top).
struct LiveVideoLayer: View {
    @ObservedObject var controller: LiveController
    var videoGravity: AVLayerVideoGravity = .resizeAspectFill

    var body: some View {
        switch controller.mode {
        case .whep:
            if let track = controller.whepTrack {
                RTCVideoView(track: track, contentMode: contentMode)
            }
        case .hls, .none:
            if let player = controller.hlsPlayer {
                PlayerLayerView(player: player, videoGravity: videoGravity)
            }
        }
    }

    private var contentMode: UIView.ContentMode {
        videoGravity == .resizeAspect ? .scaleAspectFit : .scaleAspectFill
    }
}
