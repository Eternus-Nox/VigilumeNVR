import AVFoundation
import SwiftUI
import WebRTC

/// Full-screen (landscape-friendly) live player. SMALL-RUNG FIRST: opens on the
/// `<name>_sub` substream so a frame appears quickly, then `LiveController`
/// promotes to the full-res `<name>` stream once measured link quality proves it
/// can carry it (and demotes again if it can't). Beneath that sits the older HLS
/// safety net — `<name>` primary with a `<name>_sub` fallback carrying the
/// "SD (compat)" badge and an HD retry (docs/ios-design.md §2.1.1). Muted by
/// default — tapping the video (or the pill) unmutes.
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
    @ObservedObject private var network = NetworkQuality.shared

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

                ZoomableVideo(onSingleTap: toggleMute) {
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
                if camera.capabilities.mic, !camera.isPrivate {
                    muteButton
                }
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
            case .active:
                model.resume()
            default:
                break
            }
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

    private var muteButton: some View {
        Button(action: toggleMute) {
            HStack(spacing: 7) {
                Image(systemName: model.isMuted ? "speaker.slash.fill" : "speaker.wave.2.fill")
                Text(model.isMuted ? "Tap to unmute" : "Sound on")
            }
            .font(.subheadline.weight(.semibold))
            .foregroundStyle(.white)
            .padding(.horizontal, 16)
            .padding(.vertical, 9)
            .background(Capsule().fill(Color.black.opacity(0.55)))
            .overlay(Capsule().stroke(Color.white.opacity(0.15), lineWidth: 1))
        }
        .buttonStyle(.plain)
        .padding(.bottom, 24)
    }

    // MARK: Sound

    private func toggleMute() {
        // Do NOT force a playback session here. Live view is WebRTC/VPIO and
        // needs .playAndRecord, which WHEPPlayer applies from isMuted.didSet.
        // Calling LiveAudioSession.activatePlayback() first set .playback and
        // silenced it.
        model.isMuted.toggle()
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
