import AVFoundation
import SwiftUI
import WebRTC

/// Full-screen (landscape-friendly) live player. SMALL-RUNG FIRST: opens on the
/// `<name>_sub` substream so a frame appears quickly, then `LiveController`
/// promotes to the full-res `<name>` stream once measured link quality proves it
/// can carry it (and demotes again if it can't). Beneath that sits the older HLS
/// safety net — `<name>` primary with a `<name>_sub` fallback carrying the
/// "SD (compat)" badge and an HD retry (docs/ios-design.md §2.1.1).
///
/// SOUND IS ON from the moment it opens and there is no mute control. Going
/// fullscreen on one camera is already the deliberate act of paying attention
/// to it, and a muted-by-default player asks the user to discover a second
/// gesture before the camera can be heard. The dashboard tiles stay muted —
/// several of them playing at once is a different situation — so this sets
/// `isMuted` on its own controller rather than changing the class default.
///
/// The only mic here is HOLD TO TALK. On a PTZ camera it sits in the CENTRE of
/// the directional pad, where the thumb already is.
///
/// Camera controls live on CameraDetailView (the tap-through screen that
/// presents this cover) — this view is deliberately video-only.
struct SingleCameraView: View {
    @EnvironmentObject private var session: SessionModel
    @Environment(\.dismiss) private var dismiss
    @Environment(\.scenePhase) private var scenePhase

    let camera: Camera
    /// Small rung (`<cam>_sub`) — fullscreen opens here so it appears fast.
    let whepURL: URL?
    /// Full-res rung, promoted to once the link proves it can carry it.
    let whepHighURL: URL?
    let primaryURL: URL?
    let fallbackURL: URL?

    @StateObject private var model = LiveController(keepsScreenAwake: true)
    /// Two-way talk, so an answered doorbell call can actually be answered here
    /// rather than only on the controls screen. Same pipeline CameraDetailView
    /// drives — mic -> 8 kHz Int16 PCM -> the camera's talk WebSocket.
    @StateObject private var talk = TalkController()
    @ObservedObject private var network = NetworkQuality.shared
    /// Step magnitude for the fullscreen pad. Local to this screen — the
    /// controls screen keeps its own, and a speed slider over live video is the
    /// kind of chrome fullscreen exists to avoid.
    @State private var ptzSpeed: Double = 4
    /// Last PTZ error, surfaced in the same alert talk uses.
    @State private var ptzError: String?

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            // Aspect-fit the whole wide (16:9) scene without cropping — this
            // reads correctly in BOTH orientations: letterboxed to width in
            // portrait, and filling naturally in landscape where the 16:9 frame
            // matches the screen. The layer re-lays-out on rotation, so no
            // orientation-specific gravity is needed. (Tiles + the detail card
            // stay .resizeAspectFill — ~16:9 frames where fill barely crops.)
            // Pinch/pan to zoom (1×…4×); a single tap still toggles sound
            // (unmute on first tap), double-tap resets zoom.
            if camera.isPrivate {
                PrivacyModeFullscreenOverlay()
            } else {
                // Cached frame under the video so fullscreen opens on an image
                // instead of black. `.fit` mirrors the `.resizeAspect` gravity
                // below, so the poster→video swap doesn't jump.
                if isOnline {
                    LivePosterImage(
                        url: session.api?.cameraSnapshotURL(camera.name),
                        isPlaying: model.state == .playing,
                        contentMode: .fit
                    )
                    .ignoresSafeArea()
                }

                ZoomableVideo {
                    LiveVideoLayer(controller: model, videoGravity: .resizeAspect)
                }
                .ignoresSafeArea()

                PlayerStateOverlay(
                    state: model.state,
                    isOnline: isOnline,
                    failureText: model.failureText,
                    onRetry: { model.retryPrimary() }
                )
                .allowsHitTesting(model.failureText != nil)
            }

            VStack {
                topBar
                Spacer()
                bottomBar
            }
        }
        .preferredColorScheme(.dark)
        .onAppear {
            // Fullscreen live opts into landscape: widen the app-wide gate so
            // turning the phone sideways rotates this view (the rest of the app
            // stays portrait). We don't force a rotation here — we follow the
            // device — so an already-sideways phone lands in landscape too.
            AppDelegate.orientationLock = [.portrait, .landscapeLeft, .landscapeRight]
            applyOrientationLock(rotateTo: nil)
            // Keep the screen awake while watching fullscreen live — otherwise
            // the idle timer dims/locks the phone mid-stream.
            UIApplication.shared.isIdleTimerDisabled = true
            attach()
        }
        .onDisappear {
            // Re-lock to portrait AND actively rotate back, so dismissing while
            // the phone is sideways returns the app to portrait immediately.
            AppDelegate.orientationLock = .portrait
            applyOrientationLock(rotateTo: .portrait)
            // Restore normal auto-lock when leaving fullscreen.
            UIApplication.shared.isIdleTimerDisabled = false
            // Before model.stop(): releases the mic and the talk WS rather than
            // leaving a live uplink behind a dismissed screen.
            talk.stop()
            model.stop()
        }
        // Media base flipped (LAN ⇄ primary): the parent recomputes these URLs;
        // re-attach so the full-screen player re-points to the current host.
        .onChange(of: camera.isPrivate) { _, _ in attach() }
        .onChange(of: primaryURL) { _, _ in attach() }
        .onChange(of: whepURL) { _, _ in attach() }
        .onChange(of: fallbackURL) { _, _ in attach() }
        .onChange(of: scenePhase) { _, phase in
            switch phase {
            case .background:
                model.suspend()
                talk.stop()   // release mic + WS cleanly, don't let iOS kill it
            case .active:
                model.resume()
            default:
                break
            }
        }
        // Talk is only a conversation if the far end is audible: force the live
        // receive audio on while talking, and restore the prior state after.
        .onChange(of: talk.state) { _, talkState in
            handleTalkAudio(talkState)
        }
        .alert(
            "Talk",
            isPresented: Binding(
                get: { talk.alertMessage != nil },
                set: { if !$0 { talk.alertMessage = nil } }
            )
        ) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(talk.alertMessage ?? "")
        }
        // A SECOND alert rather than one shared with talk: they can fail
        // independently (holding the mic while stepping the camera), and one
        // binding would let the later failure silently replace the earlier.
        .alert(
            "Camera control",
            isPresented: Binding(
                get: { ptzError != nil },
                set: { if !$0 { ptzError = nil } }
            )
        ) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(ptzError ?? "")
        }
    }

    private var isOnline: Bool {
        session.cameraOnline[camera.name] ?? camera.online
    }

    /// Push the current `AppDelegate.orientationLock` to the active window scene
    /// (iOS 17 geometry API). Pass `rotateTo` to also force a rotation now;
    /// pass nil to just refresh the allowed set and let the device drive it.
    private func applyOrientationLock(rotateTo: UIInterfaceOrientationMask?) {
        guard let scene = UIApplication.shared.connectedScenes
            .compactMap({ $0 as? UIWindowScene })
            .first(where: { $0.activationState == .foregroundActive })
        else { return }
        scene.keyWindow?.rootViewController?.setNeedsUpdateOfSupportedInterfaceOrientations()
        if let rotateTo {
            scene.requestGeometryUpdate(.iOS(interfaceOrientations: rotateTo)) { _ in }
        }
    }

    private func attach() {
        // Privacy Mode: the backend serves no stream for this camera, so
        // attaching would only spin and fail. Bail before play() — the overlay
        // below explains the state instead. Cosmetic only; the real
        // enforcement is server-side and never trusts this client.
        guard !camera.isPrivate else {
            model.stop()
            return
        }
        // Sound on, always, for this screen only (see the type docstring).
        // Set BEFORE play() so the first negotiated track already carries audio
        // rather than needing a second pass to enable it.
        model.isMuted = false
        guard let primaryURL else { return }
        // Open on the SMALL rung regardless of the path, then let measured
        // link quality climb to main. Starting full-res on a weak link is what
        // produced the long black open; `preferHD` only ever reached the HLS
        // fallback leg, never the WebRTC attempt.
        model.play(
            whepLow: whepURL,
            whepHigh: whepHighURL,
            primary: primaryURL,
            fallback: fallbackURL,
            preferHD: !network.isConstrained
        )
    }

    // MARK: Chrome

    private var topBar: some View {
        HStack(spacing: 12) {
            Button {
                dismiss()
            } label: {
                Image(systemName: "xmark")
                    .font(.body.weight(.semibold))
                    .foregroundStyle(.white)
                    .frame(width: 36, height: 36)
                    .background(Circle().fill(Color.black.opacity(0.5)))
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Close")

            VStack(alignment: .leading, spacing: 1) {
                Text(camera.friendlyName)
                    .font(.headline)
                    .foregroundStyle(.white)
                    .lineLimit(1)
                HStack(spacing: 5) {
                    Circle()
                        .fill(isOnline ? Theme.success : Theme.danger)
                        .frame(width: 7, height: 7)
                    Text(isOnline ? "Online" : "Offline")
                        .font(.caption)
                        .foregroundStyle(Color.white.opacity(0.75))
                }
            }

            Spacer()

            if model.usingFallback {
                SDCompatBadge { model.retryPrimary() }
            }
            if model.state == .playing {
                LiveBadge()
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(
            LinearGradient(
                colors: [Color.black.opacity(0.55), .clear],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea(edges: .top)
        )
    }

    /// The hold-to-talk mic — the only mic on this screen — and, on a PTZ
    /// camera, the directional pad it sits in the middle of.
    ///
    /// Talk is capability-gated on `speaker` (whether the camera can play what
    /// you say), NOT on `mic` (whether there is anything to hear). Plenty of
    /// cameras have one and not the other, and gating talk on `mic` would offer
    /// a button that could only ever fail.
    ///
    /// Talk is capability-gated ONLY — viewers may talk, matching the backend's
    /// talk WS, which accepts a viewer session token (see CameraDetailView).
    private var bottomBar: some View {
        VStack(spacing: 10) {
            if showsPTZ {
                // Mic in the CENTRE of the pad: on a camera you can both drive
                // and talk through, the thumb is already on the pad, and a
                // separate mic elsewhere on screen means moving off it mid-shot.
                // Compact hides the speed slider and presets — those belong on
                // the controls screen, not over live video.
                PTZControlsView(
                    onStep: { direction in Task { await ptzStep(direction) } },
                    onPresetGoto: { _ in },
                    onPresetSet: { _ in },
                    onPresetClear: { _ in },
                    savedPresets: [],
                    presetBusy: nil,
                    speed: $ptzSpeed,
                    enabled: isOnline,
                    compact: true
                ) {
                    if showsTalk {
                        talkButton(diameter: 60)
                    } else {
                        // A PTZ camera with no speaker has nothing to put here;
                        // the placeholder keeps the 3x3 grid reading as a grid.
                        PTZCenterPlaceholder()
                    }
                }
                .padding(12)
                .background(
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .fill(Color.black.opacity(0.42))
                )
            } else if showsTalk {
                talkButton(diameter: 76)
            }

            if showsTalk {
                Text(talkStatusText)
                    .font(.caption2.weight(.medium))
                    .foregroundStyle(talk.state == .live ? Theme.success : Color.white.opacity(0.75))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 3)
                    .background(Capsule().fill(Color.black.opacity(0.5)))
            }
        }
        .padding(.bottom, 24)
    }

    private func talkButton(diameter: CGFloat) -> some View {
        PushToTalkButton(model: talk, state: \.state, diameter: diameter) {
            guard let url = session.api?.talkWebSocketURL(camera: camera.name) else {
                talk.alertMessage = "Talk connection URL is unavailable."
                return
            }
            talk.start(url: url, protocols: session.api?.wsSubprotocols() ?? [])
        } onRelease: {
            talk.stop()
        }
        .disabled(!isOnline)
        .opacity(isOnline ? 1 : 0.55)
        .accessibilityLabel("Hold to talk")
    }

    /// The pad only appears for a camera that actually has PTZ, and never on a
    /// private one — a camera capturing nothing should not be drivable from a
    /// screen showing nothing.
    private var showsPTZ: Bool {
        camera.capabilities.ptz && !camera.isPrivate
    }

    /// One PTZ step. Errors surface through the same alert talk uses rather
    /// than a silent no-op — a pad that does nothing is indistinguishable from
    /// a camera that will not move.
    private func ptzStep(_ direction: PTZDirection) async {
        guard let api = session.api else { return }
        do {
            try await api.ptz(
                camera: camera.name,
                action: .step,
                direction: direction,
                speed: Int(ptzSpeed.rounded())
            )
        } catch {
            session.handleAPIError(error)
            ptzError = (error as? ApiError)?.message ?? error.localizedDescription
        }
    }

    /// Talk needs a speaker on the camera, and is pointless on a private one
    /// (capture is off server-side, so there is nothing to talk into).
    private var showsTalk: Bool {
        camera.capabilities.speaker && !camera.isPrivate
    }

    private var talkStatusText: String {
        if !isOnline { return "Camera offline" }
        switch talk.state {
        case .idle: return "Hold to talk"
        case .connecting: return "Connecting…"
        case .live: return "Talking — release to stop"
        }
    }

    /// Talk is only a conversation if the far end is audible. With no mute
    /// control this is now belt-and-braces rather than a state machine — there
    /// is nothing that could have muted it — but a talk session that goes
    /// one-way because some other path silenced playout is a bad failure, and
    /// re-asserting it costs nothing.
    ///
    /// Gated on the camera having a mic: with no camera microphone there is no
    /// receive audio to unmute, and touching `isMuted` would reconfigure the
    /// audio session for nothing mid-talk.
    private func handleTalkAudio(_ talkState: TalkController.State) {
        guard camera.capabilities.mic else { return }
        switch talkState {
        case .connecting, .live:
            if model.isMuted { model.isMuted = false }
        case .idle:
            break
        }
    }

}

/// Shared audio-session helper — activate playout so camera audio is heard even
/// with the ringer switch silenced (used by the inline card, this cover, and
/// the full-screen event clip).
///
/// **Why this drives `RTCAudioSession`, not a bare `AVAudioSession`.** The live
/// receive audio is rendered by WebRTC's own audio unit (ADM), which only
/// observes session state changed through its `RTCAudioSession` wrapper — the
/// wrapper owns an activation count and re-applies `RTCAudioSessionConfiguration
/// .webRTC()` on activate. A bare `AVAudioSession.setCategory/​setActive` (what
/// this used to do) is INVISIBLE to that wrapper: it reconfigures the underlying
/// session out from under the running audio unit, WebRTC never re-activates, and
/// playout dies — so unmuting produced no sound on every camera. Routing through
/// `RTCAudioSession` keeps WebRTC's unit in sync (fixing receive audio) while
/// still forcing `.playback` for the HLS-fallback `AVPlayer` path.
enum LiveAudioSession {
    /// Activate a PLAYBACK-only session for AVPlayer content (event clips): no
    /// mic, no voice processing. **Live view must NOT use this** — WHEPPlayer
    /// owns the live session and needs `.playAndRecord`.
    ///
    /// CRITICAL: `RTCAudioSessionConfiguration.webRTC()` returns the SHARED
    /// GLOBAL config OBJECT, not a copy. This used to fetch and MUTATE it,
    /// permanently rewriting the category WebRTC re-applies on every one of its
    /// own (re)activations — turning live view's required `.playAndRecord` into
    /// `.playback` app-wide. `.playback` silences VoiceProcessingIO: no mic and,
    /// because VPIO is full-duplex, no playout either. That is why live audio
    /// died (and stayed dead until relaunch) after unmuting or viewing a clip.
    /// Build a SEPARATE config object and leave the global one alone: READ the
    /// shared config's known-good numeric params (sample rate / IO buffer /
    /// channels), then override only the category bits for playback.
    ///
    /// Do NOT rely on a bare `RTCAudioSessionConfiguration()`'s defaults for the
    /// numeric fields — WebRTC ships this as a binary framework, so `init()`'s
    /// values aren't visible/guaranteed here, and applying zeroed sample-rate /
    /// buffer values silences AVPlayer clips.
    static func activatePlayback() {
        let session = RTCAudioSession.sharedInstance()
        session.lockForConfiguration()
        defer { session.unlockForConfiguration() }

        let shared = RTCAudioSessionConfiguration.webRTC()  // READ ONLY — never mutate
        let config = RTCAudioSessionConfiguration()
        config.sampleRate = shared.sampleRate
        config.ioBufferDuration = shared.ioBufferDuration
        config.inputNumberOfChannels = shared.inputNumberOfChannels
        config.outputNumberOfChannels = shared.outputNumberOfChannels
        config.category = AVAudioSession.Category.playback.rawValue
        config.mode = AVAudioSession.Mode.moviePlayback.rawValue
        config.categoryOptions = [.mixWithOthers]
        try? session.setConfiguration(config, active: true)
    }

    /// Release a session taken by `activatePlayback()`. RTCAudioSession
    /// reference-counts, so EVERY activatePlayback must be balanced by exactly
    /// one of these on teardown — otherwise the count never returns to zero, the
    /// session stays active, and the mic-scoping (which relies on the count
    /// reaching zero) silently stops working.
    static func deactivate() {
        let session = RTCAudioSession.sharedInstance()
        session.lockForConfiguration()
        try? session.setActive(false)
        session.unlockForConfiguration()
    }
}

/// Fullscreen "Privacy Mode" treatment, the counterpart to the dashboard tile's
/// `PrivacyModeTileOverlay`. Shown instead of the player (and instead of the
/// retry-able failure overlay) so a deliberately-off camera never reads as a
/// broken stream the user should try to fix.
private struct PrivacyModeFullscreenOverlay: View {
    var body: some View {
        VStack(spacing: 10) {
            Image(systemName: "eye.slash.fill")
                .font(.system(size: 40))
                .foregroundStyle(Color.white.opacity(0.85))
            Text("Privacy Mode")
                .font(.headline)
                .foregroundStyle(Color.white.opacity(0.9))
            Text("This camera is not recording or streaming.")
                .font(.footnote)
                .foregroundStyle(Color.white.opacity(0.6))
                .multilineTextAlignment(.center)
        }
        .padding(24)
        .accessibilityElement(children: .combine)
    }
}
