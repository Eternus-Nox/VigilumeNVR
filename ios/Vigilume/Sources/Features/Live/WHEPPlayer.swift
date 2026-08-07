import AVFoundation
import Combine
import Foundation
import SwiftUI
import WebRTC

/// One receive-only WebRTC peer connection that pulls a go2rtc stream over WHEP
/// (docs/ios-design.md §2) — the sub-second live path that replaces HLS on
/// networks where WebRTC can establish.
///
/// **Signaling (verified against go2rtc v1.9.14 source).** go2rtc's only WebRTC
/// HTTP route is `api/webrtc` (proxied at `/go2rtc/api/webrtc?src=<name>`). We
/// build a recvonly `video`+`audio` unified-plan offer, gather host ICE
/// candidates (non-trickle: WHEP is a single POST, so the offer we send already
/// carries them), then `POST` the raw offer SDP with `Content-Type:
/// application/sdp`. go2rtc's `outputWebRTC` returns `201 Created` with the
/// answer SDP as the body, which we apply as the remote description. On LAN the
/// server auto-advertises its host candidates (handled by the backend
/// workflow), so the connection completes without any STUN/TURN.
///
/// **"Playing" means real frames.** `state` only becomes `.playing` once the
/// decoder emits a first video frame (a hidden `FrameSink` renderer on the
/// track), not merely when ICE connects. So a peer that connects but can't
/// decode the video (e.g. an HEVC main that go2rtc won't transcode) never
/// reports success — which is exactly what lets `LiveController` fall back to
/// HLS instead of sitting on a black WebRTC view.
///
/// **Teardown is leak-free.** `stop()` cancels the in-flight signaling task,
/// detaches the frame sink, disables + drops the tracks, and closes the peer
/// connection. Nothing survives a `disappear`/background.
/// One reading of how the inbound video is ACTUALLY arriving, reduced from an
/// `RTCStatisticsReport`. This is the only honest signal we have about link
/// quality, because nothing in this pipeline can adapt bitrate: go2rtc repacks
/// the camera's already-encoded H.264 (no transcode, no simulcast, no SVC), so
/// an over-capacity link does NOT ramp down — it freezes and drops packets.
/// Those symptoms are exactly what this measures, and `LiveController` turns
/// them into a switch to the smaller `_sub` stream.
struct LiveQualitySample: Sendable, Equatable {
    /// Cumulative counters straight from inbound-rtp (deltas are what matter).
    let framesDecoded: Int
    let freezeCount: Int
    let packetsLost: Int
    let bytesReceived: Int
    /// Instantaneous decode rate as reported by WebRTC (0 when absent).
    let framesPerSecond: Double
    let jitter: Double
    /// Monotonic capture time, so callers can compute real per-second rates.
    let at: TimeInterval
}

@MainActor
final class WHEPPlayer: NSObject, ObservableObject {
    enum State: Equatable { case idle, connecting, playing, failed }

    @Published private(set) var state: State = .idle
    /// Latest inbound-video stats (nil until the first sample lands). Read-only
    /// telemetry — sampling never touches playback.
    @Published private(set) var quality: LiveQualitySample?
    /// The remote video track, ready to attach to an `RTCMTLVideoView`. Set as
    /// soon as the recvonly transceiver exists; it renders once frames arrive.
    @Published private(set) var videoTrack: RTCVideoTrack?

    /// Mirrors the shared mute state onto the remote audio track (disabled while
    /// muted so WebRTC never opens the speaker for a silent tile).
    var isMuted: Bool = true {
        didSet {
            let wantAudio = !isMuted
            audioTrack?.isEnabled = wantAudio
            // Unmute: apply + activate the .playAndRecord/.voiceChat session so
            // VPIO opens and plays the camera's audio out of the speaker.
            // Mute: release it, so the mic (and its iOS indicator) are held only
            // while actually listening — never by a muted view or the grid.
            if wantAudio {
                activateAudioIfNeeded()
            } else {
                releaseAudioIfNeeded()
            }
        }
    }

    /// Apply the WebRTC audio config to the live AVAudioSession and activate it.
    /// Called on unmute.
    ///
    /// DO NOT reintroduce `useManualAudio` here. Scoping the mic indicator that
    /// way (isAudioEnabled toggled per mute) left the session inactive, so the
    /// VoiceProcessingIO unit never started — no mic AND, since VPIO is
    /// full-duplex, no playout either: totally silent live view. WebRTC manages
    /// its own audio unit; we only make sure the session is configured + active.
    /// The cost is the iOS mic-in-use indicator during live view — a known,
    /// accepted trade-off for audio that actually works.
    static func activateAudioSession() {
        let session = RTCAudioSession.sharedInstance()
        session.lockForConfiguration()
        do {
            try session.setConfiguration(RTCAudioSessionConfiguration.webRTC(), active: true)
        } catch {
            // Never crash over audio — the tile just stays silent.
        }
        session.unlockForConfiguration()
    }

    /// Release the audio session, so the mic (and its iOS indicator) are held
    /// ONLY while a camera view is actually listening — not app-wide while you
    /// browse the grid. Called from stop(), i.e. leaving a camera, suspending,
    /// or the teardown half of a switch.
    ///
    /// Re-activation is covered on BOTH paths back in: unmuting (isMuted.didSet)
    /// and a new connection binding its audio track (connect()). That pairing is
    /// what makes deactivating here safe — scoping the mic by flipping
    /// `useManualAudio` instead silenced live view twice (see
    /// activateAudioSession).
    static func deactivateAudioSession() {
        let session = RTCAudioSession.sharedInstance()
        session.lockForConfiguration()
        // Throws if something else still holds the session — never fatal; the
        // worst case is it lingers until the next teardown.
        try? session.setActive(false)
        session.unlockForConfiguration()
    }

    /// Has THIS player activated the shared session? RTCAudioSession
    /// reference-counts activations and only really deactivates on the last
    /// balanced call — so every activate must be matched by exactly ONE release,
    /// and a player that never activated must never release.
    ///
    /// Unbalanced calls are not a tidiness issue, they silence other screens: an
    /// unmatched `setActive(false)` from a live player tearing down dropped the
    /// count to zero and killed the session an event clip / the timeline had
    /// just activated (SwiftUI runs the incoming view's onAppear BEFORE the
    /// outgoing view's onDisappear). Balanced, the count stays >0 and their audio
    /// survives.
    private var didActivateAudio = false

    private func activateAudioIfNeeded() {
        guard !didActivateAudio else { return }
        Self.activateAudioSession()
        didActivateAudio = true
    }

    private func releaseAudioIfNeeded() {
        guard didActivateAudio else { return }
        Self.deactivateAudioSession()
        didActivateAudio = false
    }

    private var peerConnection: RTCPeerConnection?
    private var audioTrack: RTCAudioTrack?
    private var frameSink: FrameSink?
    private var connectTask: Task<Void, Never>?
    private var didSignalFirstFrame = false
    /// Periodic inbound-video stats poll; runs only while a frame is flowing.
    private var statsTask: Task<Void, Never>?
    /// How often to sample. Slow on purpose: this feeds a decision that is
    /// deliberately sluggish (see LiveController's demote/promote windows), and
    /// getStats is not free.
    private static let statsInterval: UInt64 = 2_000_000_000  // 2 s
    /// The single in-flight ICE-gathering waiter, if any. Main-actor isolated
    /// (the whole class is), which is what makes the "resume exactly once"
    /// guarantee in `resumeICEGatheringWaiter` real — a second `resume` on the
    /// same continuation is a runtime trap, and both the delegate and the
    /// timeout arm can reach it.
    private var iceGatheringContinuation: CheckedContinuation<Void, Never>?

    // MARK: Shared, expensive-to-build WebRTC singletons

    /// Configure WebRTC's audio session ONCE, before any peer connection exists,
    /// so the recvonly viewer can actually play the camera's audio on iOS (see
    /// the category rationale inline below).
    private static let configureAudioSession: Void = {
        let config = RTCAudioSessionConfiguration.webRTC()
        // WebRTC's default audio unit is VoiceProcessingIO (full-duplex): it will
        // NOT output received audio unless the session ALSO grants the mic side
        // (.playAndRecord). With the .playback category it stayed SILENT on iOS
        // (web worked). This WebRTC build has no bypassVoiceProcessing option, so
        // .playAndRecord is required; .defaultToSpeaker routes to the loud
        // speaker (not the earpiece), .voiceChat is VPIO's mode. Trade-off: iOS
        // shows the mic-in-use indicator during live view even though this path
        // never sends audio (two-way talk is a separate TalkController pipeline).
        config.category = AVAudioSession.Category.playAndRecord.rawValue
        config.mode = AVAudioSession.Mode.voiceChat.rawValue
        // NO .mixWithOthers: listening to a camera should TAKE OVER audio, not
        // play under someone's music. Without it, activating this session
        // interrupts other apps (they stop); deactivating on mute / leaving the
        // camera lets them resume. .defaultToSpeaker keeps output on the loud
        // speaker rather than the earpiece (a .playAndRecord default).
        config.categoryOptions = [.defaultToSpeaker]
        RTCAudioSessionConfiguration.setWebRTC(config)
        // NOTE: deliberately NOT using RTCAudioSession.useManualAudio. Letting
        // WebRTC own its audio unit is what makes playout work; manual mode
        // silenced live view entirely (see activateAudioSession).
    }()

    /// Whether this player may EVER play audio. False for grid tiles, which are
    /// permanently muted.
    ///
    /// This gates whether we negotiate an audio transceiver at all — and that
    /// matters far beyond bandwidth. WebRTC (correctly, since we don't use
    /// manual audio) auto-starts its audio unit for ANY connection carrying
    /// audio, and our session is `.playAndRecord`, so that unit opens the MIC.
    /// Muting only disables the TRACK, not the negotiation — so a wall of muted
    /// tiles still lit the mic indicator the moment they connected, even with
    /// nothing listening.
    private let allowsAudio: Bool

    init(allowsAudio: Bool = true) {
        self.allowsAudio = allowsAudio
        _ = Self.configureAudioSession
        super.init()
    }

    deinit {
        connectTask?.cancel()
        // Delegate callbacks are gone once the PC is closed; safe from deinit.
        peerConnection?.close()
    }

    // MARK: Lifecycle

    /// Start (or restart) a WHEP session against `url` (already carrying
    /// `?src=<stream>`). Safe to call repeatedly; it tears down any prior
    /// session first. The offer is strictly receive-only (recvonly video +
    /// recvonly audio) for EVERY camera — two-way talk rides the separate
    /// `TalkController` WS pipeline (never this peer connection) so the receive
    /// path negotiates cleanly.
    func start(url: URL) {
        stop()
        state = .connecting
        connectTask = Task { [weak self] in
            await self?.connect(url: url)
        }
    }

    /// Full teardown: cancel signaling, detach the sink, drop tracks, close PC.
    func stop() {
        connectTask?.cancel()
        connectTask = nil
        statsTask?.cancel()
        statsTask = nil
        quality = nil
        didSignalFirstFrame = false
        // The cancelled signaling task may be parked on ICE gathering; the peer
        // connection is about to close, so no delegate callback is coming. Wake
        // it now (it bails on `Task.isCancelled` immediately after) rather than
        // leaving it to sit out the 1.5 s cap.
        resumeICEGatheringWaiter()

        if let frameSink { videoTrack?.remove(frameSink) }
        frameSink = nil
        videoTrack?.isEnabled = false
        audioTrack?.isEnabled = false
        // Leaving the camera (or the teardown half of a switch): release the
        // session so the mic + its iOS indicator are scoped to actively viewing
        // a camera, and are NOT held while you browse the grid. Coming back in
        // re-activates on either path: unmute (isMuted.didSet) or the new
        // connection binding its audio track (connect()). Balanced: only
        // releases if THIS player activated, so tearing a live view down can't
        // silence a clip/timeline that just activated its own session.
        releaseAudioIfNeeded()
        videoTrack = nil
        audioTrack = nil

        peerConnection?.close()
        peerConnection = nil

        if state != .idle { state = .idle }
    }

    // MARK: Signaling

    private func connect(url: URL) async {
        let config = RTCConfiguration()
        config.sdpSemantics = .unifiedPlan
        config.continualGatheringPolicy = .gatherOnce
        // LAN-only: rely on host candidates. No STUN/TURN — the server
        // auto-advertises its own host candidates in the answer.
        config.iceServers = []

        let constraints = RTCMediaConstraints(mandatoryConstraints: nil, optionalConstraints: nil)
        guard let pc = WebRTCFactory.shared.peerConnection(with: config, constraints: constraints, delegate: self) else {
            fail()
            return
        }
        peerConnection = pc

        // Recvonly transceivers — their receiver tracks exist immediately, so we
        // can bind the renderer before a single byte of media flows.
        let recvOnly = RTCRtpTransceiverInit()
        recvOnly.direction = .recvOnly
        let videoTransceiver = pc.addTransceiver(of: .video, init: recvOnly)
        // Audio ONLY for players that may actually play it (see allowsAudio) —
        // negotiating it on a muted tile makes WebRTC open the audio unit, and
        // with .playAndRecord that means the mic.
        let audioTransceiver = allowsAudio ? pc.addTransceiver(of: .audio, init: recvOnly) : nil

        if let track = videoTransceiver?.receiver.track as? RTCVideoTrack {
            let sink = FrameSink { [weak self] in self?.handleFirstFrame() }
            track.add(sink)
            frameSink = sink
            videoTrack = track
        }
        if let track = audioTransceiver?.receiver.track as? RTCAudioTrack {
            track.isEnabled = !isMuted
            audioTrack = track
            // Re-assert the audio session HERE, now that this connection's track
            // exists — not only from isMuted.didSet.
            //
            // Switching cameras runs `isMuted = …` (didSet activates) and THEN
            // start() -> stop(), which closes the previous peer connection and
            // takes WebRTC's audio unit down with it. So the activation happened
            // BEFORE this connection existed and the new one comes up on a dead
            // session: no mic, and (VPIO being full-duplex) no playout either —
            // the first camera worked, every switch after it was silent.
            if !isMuted { activateAudioIfNeeded() }
        }

        // createOffer → setLocalDescription → gather host candidates.
        guard let offer = await makeOffer(pc, constraints: constraints) else { fail(); return }
        guard await setLocalDescription(pc, offer) else { fail(); return }
        await waitForICEGathering(pc)
        if Task.isCancelled { return }

        // The local description now carries the gathered candidates.
        let offerSDP = pc.localDescription?.sdp ?? offer.sdp

        // WHEP POST: raw SDP offer, application/sdp → 201 + answer SDP.
        let answerSDP: String
        do {
            answerSDP = try await postOffer(offerSDP, to: url)
        } catch {
            if !Task.isCancelled { fail() }
            return
        }
        if Task.isCancelled { return }

        let answer = RTCSessionDescription(type: .answer, sdp: answerSDP)
        guard await setRemoteDescription(pc, answer) else { fail(); return }
        // From here `didChange` connection-state callbacks + the first frame
        // drive `state`.
    }

    private func postOffer(_ sdp: String, to url: URL) async throws -> String {
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/sdp", forHTTPHeaderField: "Content-Type")
        request.setValue("application/sdp", forHTTPHeaderField: "Accept")
        request.httpBody = sdp.data(using: .utf8)
        request.timeoutInterval = 8

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse,
              (200...299).contains(http.statusCode),
              let body = String(data: data, encoding: .utf8),
              !body.isEmpty else {
            throw URLError(.badServerResponse)
        }
        return body
    }

    // MARK: Async wrappers over WebRTC's callback API

    private func makeOffer(_ pc: RTCPeerConnection, constraints: RTCMediaConstraints) async -> RTCSessionDescription? {
        await withCheckedContinuation { cont in
            pc.offer(for: constraints) { sdp, _ in cont.resume(returning: sdp) }
        }
    }

    private func setLocalDescription(_ pc: RTCPeerConnection, _ sdp: RTCSessionDescription) async -> Bool {
        await withCheckedContinuation { cont in
            pc.setLocalDescription(sdp) { error in cont.resume(returning: error == nil) }
        }
    }

    private func setRemoteDescription(_ pc: RTCPeerConnection, _ sdp: RTCSessionDescription) async -> Bool {
        await withCheckedContinuation { cont in
            pc.setRemoteDescription(sdp) { error in cont.resume(returning: error == nil) }
        }
    }

    /// Wait for ICE gathering to complete (host candidates only ⇒ fast). Capped
    /// so a stuck gather never blocks the POST for more than ~1.5 s.
    ///
    /// Event-driven, not polled: `didChange RTCIceGatheringState` resumes us the
    /// instant gathering finishes. Polling every 50 ms burned ~50-100 ms on
    /// EVERY attach (every tile, every scroll reveal, every screen hop) because
    /// with `iceServers = []` + `.gatherOnce` gathering completes well inside
    /// the first sleep.
    private func waitForICEGathering(_ pc: RTCPeerConnection) async {
        // Common case: already done, so never suspend at all.
        if pc.iceGatheringState == .complete { return }
        // Defensive: a previous waiter should be impossible (one connect task
        // at a time), but never strand one.
        resumeICEGatheringWaiter()

        // The cap. Unstructured, so it isn't cancelled out from under us; it is
        // cancelled explicitly below once we're through.
        let timeout = Task { @MainActor [weak self] in
            try? await Task.sleep(nanoseconds: 1_500_000_000)  // 1.5 s
            self?.resumeICEGatheringWaiter()
        }
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            // Re-check now that we're installing the waiter: gathering may have
            // completed between the check above and here, in which case the
            // delegate has already fired and would never fire again.
            if pc.iceGatheringState == .complete {
                cont.resume()
            } else {
                iceGatheringContinuation = cont
            }
        }
        timeout.cancel()
    }

    /// Resume the pending ICE-gathering waiter at most once. The delegate, the
    /// 1.5 s cap and `stop()` all race to get here; clearing the stored
    /// continuation before resuming makes every loser a no-op.
    private func resumeICEGatheringWaiter() {
        guard let cont = iceGatheringContinuation else { return }
        iceGatheringContinuation = nil
        cont.resume()
    }

    // MARK: State transitions

    private func handleFirstFrame() {
        guard !didSignalFirstFrame else { return }
        didSignalFirstFrame = true
        if state != .playing { state = .playing }
        startStatsSampling()
    }

    // MARK: Inbound-video stats (telemetry for the quality switcher)

    /// Poll `inbound-rtp` video stats every `statsInterval` while media flows.
    /// Started on the first decoded frame (there is nothing to measure before
    /// that) and cancelled in `stop()`.
    private func startStatsSampling() {
        statsTask?.cancel()
        statsTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: Self.statsInterval)
                if Task.isCancelled { return }
                guard let self else { return }
                await self.sampleStats()
            }
        }
    }

    /// One getStats pass, reduced to a `LiveQualitySample`. Never throws, never
    /// affects playback: a missing report or field just skips the sample.
    private func sampleStats() async {
        guard let pc = peerConnection else { return }
        let report: RTCStatisticsReport = await withCheckedContinuation { cont in
            // The completion handler fires on a WebRTC thread; the continuation
            // hops us back to the main actor (this method is main-actor bound).
            pc.statistics { cont.resume(returning: $0) }
        }
        guard !Task.isCancelled else { return }

        // The inbound VIDEO stream. `kind` is the modern key; older builds used
        // `mediaType`, so accept either rather than silently sampling nothing.
        let inboundVideo = report.statistics.values.first { stat in
            guard stat.type == "inbound-rtp" else { return false }
            let kind = (stat.values["kind"] as? String) ?? (stat.values["mediaType"] as? String)
            return kind == "video"
        }
        guard let v = inboundVideo?.values else { return }

        func int(_ key: String) -> Int { (v[key] as? NSNumber)?.intValue ?? 0 }
        func dbl(_ key: String) -> Double { (v[key] as? NSNumber)?.doubleValue ?? 0 }

        quality = LiveQualitySample(
            framesDecoded: int("framesDecoded"),
            freezeCount: int("freezeCount"),
            packetsLost: int("packetsLost"),
            bytesReceived: int("bytesReceived"),
            framesPerSecond: dbl("framesPerSecond"),
            jitter: dbl("jitter"),
            at: ProcessInfo.processInfo.systemUptime
        )
    }

    private func fail() {
        if state != .failed { state = .failed }
    }

    fileprivate func connectionStateChanged(_ newState: RTCPeerConnectionState) {
        switch newState {
        case .connected:
            // Frames confirm true playback; connection alone doesn't flip to
            // .playing. But if we're already playing, stay playing.
            break
        case .failed, .closed:
            fail()
        case .disconnected:
            // A transient blip may recover; only treat a hard failure/close as
            // fatal. If it never recovers the connection ultimately goes
            // .failed and we fall back then.
            break
        default:
            break
        }
    }
}

// MARK: - RTCPeerConnectionDelegate (all callbacks hop to the main actor)

extension WHEPPlayer: RTCPeerConnectionDelegate {
    nonisolated func peerConnection(_ peerConnection: RTCPeerConnection, didChange newState: RTCPeerConnectionState) {
        Task { @MainActor [weak self] in self?.connectionStateChanged(newState) }
    }

    nonisolated func peerConnection(_ peerConnection: RTCPeerConnection, didChange stateChanged: RTCSignalingState) {}
    nonisolated func peerConnection(_ peerConnection: RTCPeerConnection, didAdd stream: RTCMediaStream) {}
    nonisolated func peerConnection(_ peerConnection: RTCPeerConnection, didRemove stream: RTCMediaStream) {}
    nonisolated func peerConnectionShouldNegotiate(_ peerConnection: RTCPeerConnection) {}
    nonisolated func peerConnection(_ peerConnection: RTCPeerConnection, didChange newState: RTCIceConnectionState) {
        if newState == .failed {
            Task { @MainActor [weak self] in self?.fail() }
        }
    }
    nonisolated func peerConnection(_ peerConnection: RTCPeerConnection, didChange newState: RTCIceGatheringState) {
        // Wakes `waitForICEGathering` the moment gathering finishes, instead of
        // it polling. Harmless if nothing is waiting.
        guard newState == .complete else { return }
        Task { @MainActor [weak self] in self?.resumeICEGatheringWaiter() }
    }
    nonisolated func peerConnection(_ peerConnection: RTCPeerConnection, didGenerate candidate: RTCIceCandidate) {}
    nonisolated func peerConnection(_ peerConnection: RTCPeerConnection, didRemove candidates: [RTCIceCandidate]) {}
    nonisolated func peerConnection(_ peerConnection: RTCPeerConnection, didOpen dataChannel: RTCDataChannel) {}
}

// MARK: - First-frame detector

/// A near-no-op `RTCVideoRenderer` attached alongside the visible view purely to
/// learn when the decoder produces its first real frame. Called on a WebRTC
/// worker thread — the closure hops to the main actor itself.
private final class FrameSink: NSObject, RTCVideoRenderer {
    private let onFirstFrame: () -> Void
    private var fired = false

    init(onFirstFrame: @escaping () -> Void) {
        self.onFirstFrame = onFirstFrame
    }

    func setSize(_ size: CGSize) {}

    func renderFrame(_ frame: RTCVideoFrame?) {
        guard frame != nil, !fired else { return }
        fired = true
        let callback = onFirstFrame
        Task { @MainActor in callback() }
    }
}

// MARK: - SwiftUI wrapper around RTCMTLVideoView

/// Renders an `RTCVideoTrack` in a Metal-backed view. `videoContentMode`
/// mirrors AVLayerVideoGravity: `.scaleAspectFill` for tiles/full-screen (crop
/// to fill, matching `PlayerLayerView`'s `.resizeAspectFill`).
struct RTCVideoView: UIViewRepresentable {
    let track: RTCVideoTrack?
    var contentMode: UIView.ContentMode = .scaleAspectFill

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeUIView(context: Context) -> RTCMTLVideoView {
        let view = RTCMTLVideoView()
        view.videoContentMode = contentMode
        view.clipsToBounds = true
        view.backgroundColor = .clear
        return view
    }

    func updateUIView(_ uiView: RTCMTLVideoView, context: Context) {
        uiView.videoContentMode = contentMode
        let coordinator = context.coordinator
        if coordinator.track !== track {
            coordinator.track?.remove(uiView)
            coordinator.track = track
            track?.add(uiView)
        }
    }

    static func dismantleUIView(_ uiView: RTCMTLVideoView, coordinator: Coordinator) {
        coordinator.track?.remove(uiView)
        coordinator.track = nil
    }

    final class Coordinator {
        weak var track: RTCVideoTrack?
    }
}
