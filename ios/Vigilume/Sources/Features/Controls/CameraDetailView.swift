import SwiftUI

/// THE one-camera screen — what tapping a camera tile opens. It's a
/// control-focused page: a big 16:9 live player on top (HD main stream with
/// automatic SD-substream fallback, tap-to-unmute, expand-to-fullscreen),
/// then, in this order: the PTZ pad and two-way talk (CAPABILITY-gated only —
/// viewers get both), then the admin-gated camera controls (night vision,
/// spotlight, siren, reboot), the AI-detection mode setting, and the
/// recent-events strip. Camera-AI stays a silent background
/// function: only the SETTING lives here, there is no live on-camera AI
/// indicator/pill and no device/status card.
struct CameraDetailView: View {
    let camera: Camera

    @EnvironmentObject private var session: SessionModel
    @Environment(\.scenePhase) private var scenePhase
    // Two-way talk for every speaker camera: mic → 8 kHz Int16 PCM → binary
    // frames over the WS /api/cameras/{name}/talk. The backend picks delivery
    // per camera (RTSP backchannel for the AD410, CGI postAudio otherwise).
    @StateObject private var talk = TalkController()

    // Live player (big top card + fullscreen cover)
    @StateObject private var live = LiveController(keepsScreenAwake: true)
    @State private var showFullscreen = false

    // Device settings (Amcrest state behind the controls)
    @State private var device: DeviceSettings?
    @State private var deviceLoading = false
    @State private var deviceError: String?

    // Control state mirrored into SwiftUI controls
    @State private var spotlightMode: SpotlightMode = .off
    @State private var nightVisionMode: NightVisionMode = .auto

    // PTZ (caps.ptz only): which preset is mid-request. Movement is tap-to-step
    // (a single fire-and-forget nudge), so there's no held-direction state.
    @State private var ptzPresetBusy: Int?
    // Which preset slots (1…3) hold a saved position. The dome has no
    // "list presets" CGI, so we track saved/empty locally (persisted per
    // camera) and update it as set/clear succeed — this is what drives the
    // saved-vs-empty affordance and stops a tap from recalling an unset slot.
    @State private var ptzSavedPresets: Set<Int> = []
    // Step magnitude (1–8) for the directional pad; the slider writes here.
    @State private var ptzSpeed: Double = 4
    // Transient "what just happened" caption under the pad (set/goto/clear).
    @State private var ptzFeedback: String?

    // Live receive-audio mute state captured before a talk session so it can be
    // restored when talk ends (auto-unmute while talking for a real 2-way call).
    @State private var muteBeforeTalk: Bool?

    // One-shot action state
    @State private var controlError: String?
    @State private var showRebootConfirm = false
    @State private var sirenBusy = false
    @State private var rebootBusy = false

    // Recent events
    @State private var recentEvents: [Event] = []

    private var isOnline: Bool {
        session.cameraOnline[camera.name] ?? camera.online
    }

    private var hasDeviceSettings: Bool {
        camera.capabilities.ir
            || camera.capabilities.whiteLight
            || camera.capabilities.nightVision
    }

    private var showsControls: Bool {
        session.isAdmin
    }

    /// PTZ pad + presets: capability-gated only — VIEWERS may aim a camera.
    /// Aiming is a live-viewing action (it stores nothing and is bounded by the
    /// camera's own limits), not an admin configuration change. Matches the
    /// backend, where POST /api/cameras/{name}/ptz dropped require_admin.
    private var showsPTZ: Bool {
        camera.capabilities.ptz
    }

    /// Two-way talk: capability-gated only — VIEWERS may talk (answer the door,
    /// call the dog). Matches the backend talk WS, which now accepts a viewer
    /// session token (media-scope tokens are still refused).
    private var showsTalk: Bool {
        camera.capabilities.speaker
    }

    /// A camera with BOTH PTZ and two-way audio stacks a lot of control cards
    /// (PTZ pad + presets, talk, camera controls, AI) — too much to reach
    /// without scrolling. Those go behind a segmented tab bar instead so each
    /// group is one tap away. Every other camera keeps the simple stacked list.
    private var usesControlTabs: Bool {
        showsPTZ && showsTalk
    }

    private enum ControlTab: String, CaseIterable, Identifiable {
        case ptz = "PTZ"
        case audio = "Audio"
        case camera = "Camera"
        var id: String { rawValue }
    }

    @State private var controlTab: ControlTab = .ptz

    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                livePlayerCard
                // Fixed order below the player: the PTZ pad (tied to the live
                // video) sits first for a caps.ptz dome, then two-way talk, the
                // camera controls, and the AI-detection setting.
                // Every live-interaction control is hidden in Privacy Mode:
                // PTZ, two-way talk and the camera controls all act on a camera
                // the product says is capturing nothing (and the talk WS is
                // refused server-side with close 1008 anyway).
                if !camera.isPrivate {
                    if usesControlTabs {
                        controlTabs
                    } else {
                        if showsPTZ {
                            ptzCard
                        }
                        if showsTalk {
                            talkCard
                        }
                        if showsControls {
                            controlsCard
                        }
                    }
                }
                recentEventsCard
            }
            .padding(16)
        }
        .background(Theme.bg.ignoresSafeArea())
        .navigationTitle(camera.friendlyName)
        .navigationBarTitleDisplayMode(.inline)
        // Admin-only: full per-camera settings (parity with web's Device
        // settings + edit form). A viewer never sees this control.
        .toolbar {
            if session.isAdmin {
                ToolbarItem(placement: .topBarTrailing) {
                    NavigationLink {
                        CameraSettingsView(camera: camera)
                    } label: {
                        Image(systemName: "slider.horizontal.3")
                    }
                    .accessibilityLabel("Camera settings")
                }
            }
        }
        .task {
            loadSavedPresets()
            await initialLoad()
        }
        .refreshable { await initialLoad() }
        .onAppear(perform: attachLive)
        .onDisappear {
            talk.stop()
            live.stop()
        }
        .onChange(of: isOnline) { _, online in
            if online { attachLive() } else { live.stop() }
        }
        // A LAN reachability flip rebuilds session.api with a new mediaBase;
        // re-resolve the live URLs so video re-points (LAN ⇄ primary).
        .onChange(of: session.api?.mediaBase) { _, _ in attachLive() }
        .onChange(of: camera.isPrivate) { _, _ in attachLive() }
        .onChange(of: showFullscreen) { _, fullscreen in
            // Don't hold two copies of the stream while the cover is up.
            if fullscreen { live.suspend() } else { live.resume() }
        }
        // Auto-unmute the live receive audio while talking so it's a real
        // two-way conversation; restore the prior mute state when talk ends.
        .onChange(of: talk.state) { _, talkState in
            handleTalkAudio(talkState)
        }
        .onChange(of: scenePhase) { _, phase in
            switch phase {
            case .background:
                live.suspend()
                talk.stop()             // release mic + WS cleanly, don't let iOS kill it
            case .active:
                if !showFullscreen { live.resume() }
            default:
                break
            }
        }
        .fullScreenCover(isPresented: $showFullscreen) {
            SingleCameraView(
                camera: camera,
                whepURL: session.api?.liveSubStreamWHEPURL(camera: camera.name),
                whepHighURL: session.api?.liveStreamWHEPURL(camera: camera.name),
                primaryURL: session.api?.liveStreamURL(camera: camera.name),
                fallbackURL: session.api?.liveSubStreamURL(camera: camera.name)
            )
        }
        .onReceive(session.wsMessages) { message in
            switch message {
            case .eventNew(let e), .eventUpdate(let e), .eventEnd(let e), .doorbell(let e):
                if e.camera == camera.name {
                    // An event on this camera — refresh the recent-events strip.
                    Task { await loadRecentEvents() }
                }
            default:
                break
            }
        }
        .alert(
            "Camera control",
            isPresented: Binding(
                get: { controlError != nil },
                set: { if !$0 { controlError = nil } }
            )
        ) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(controlError ?? "")
        }
        .alert(
            "Two-way talk",
            isPresented: Binding(
                get: { talk.alertMessage != nil },
                set: { if !$0 { talk.alertMessage = nil } }
            )
        ) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(talk.alertMessage ?? "")
        }
        .confirmationDialog(
            "Reboot \(camera.friendlyName)?",
            isPresented: $showRebootConfirm,
            titleVisibility: .visible
        ) {
            Button("Reboot camera", role: .destructive) {
                Task { await reboot() }
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("The camera goes dark for about a minute while it restarts.")
        }
    }

    // MARK: - Live player card (big player on top; HD → SD fallback)

    private func attachLive() {
        // Privacy Mode: the backend has DELETED this camera's go2rtc streams,
        // so every attach is guaranteed to fail and would surface as
        // "Live stream failed" + a Try again button — presenting a deliberate,
        // admin-chosen state as a camera fault. Bail; the card renders the
        // privacy panel instead. Cosmetic only — capture is stopped server-side.
        guard !camera.isPrivate else {
            live.stop()
            return
        }
        guard isOnline, let api = session.api else { return }
        // Constrained/cellular: default to SD (auto-upgrades to HD when the
        // path improves); good Wi-Fi opens HD-first as before.
        // Open on the SMALL rung and let measured link quality climb to main
        // (LiveController.evaluateQuality). This used to open full-res
        // unconditionally — `preferHD` was stored but never consulted for the
        // WHEP attempt — which is why a weak link started with a long stall.
        live.play(
            whepLow: api.liveSubStreamWHEPURL(camera: camera.name),
            whepHigh: api.liveStreamWHEPURL(camera: camera.name),
            primary: api.liveStreamURL(camera: camera.name),
            fallback: api.liveSubStreamURL(camera: camera.name),
            preferHD: !NetworkQuality.shared.isConstrained
        )
    }

    private var livePlayerCard: some View {
        VStack(spacing: 0) {
            ZStack {
                Rectangle().fill(Theme.bgDeep)

                if camera.isPrivate {
                    PrivacyModeDetailOverlay()
                } else {
                // Cached frame under the video: the card shows an image within
                // ~50-150 ms instead of a dark box while the stream negotiates.
                if isOnline {
                    LivePosterImage(
                        url: session.api?.cameraSnapshotURL(camera.name),
                        isPlaying: live.state == .playing,
                        contentMode: .fill
                    )
                }

                // Fill the 16:9 card rather than letterbox a 4:3 substream.
                LiveVideoLayer(controller: live, videoGravity: .resizeAspectFill)

                // Tap on the video toggles sound (unmute on first tap).
                Color.clear
                    .contentShape(Rectangle())
                    .onTapGesture(perform: toggleLiveMute)

                PlayerStateOverlay(
                    state: live.state,
                    isOnline: isOnline,
                    failureText: live.failureText,
                    onRetry: { live.retryPrimary() }
                )
                .allowsHitTesting(live.failureText != nil)

                VStack {
                    HStack(spacing: 6) {
                        if live.usingFallback {
                            SDCompatBadge { live.retryPrimary() }
                        }
                        if live.state == .playing {
                            LiveBadge()
                        }
                        Spacer()
                        Button {
                            showFullscreen = true
                        } label: {
                            Image(systemName: "arrow.up.left.and.arrow.down.right")
                                .font(.footnote.weight(.semibold))
                                .foregroundStyle(.white)
                                .frame(width: 30, height: 30)
                                .background(Circle().fill(Color.black.opacity(0.5)))
                        }
                        .buttonStyle(.plain)
                        .accessibilityLabel("Full screen")
                    }
                    Spacer()
                    if camera.capabilities.mic, !camera.isPrivate {
                        HStack {
                            Spacer()
                            Button(action: toggleLiveMute) {
                                Image(systemName: live.isMuted
                                    ? "speaker.slash.fill" : "speaker.wave.2.fill")
                                    .font(.footnote.weight(.semibold))
                                    .foregroundStyle(.white)
                                    .frame(width: 30, height: 30)
                                    .background(Circle().fill(Color.black.opacity(0.5)))
                            }
                            .buttonStyle(.plain)
                            .accessibilityLabel(live.isMuted ? "Unmute" : "Mute")
                        }
                    }
                }
                .padding(8)
                }
            }
            .aspectRatio(16 / 9, contentMode: .fit)
            .clipped()

            HStack(spacing: 8) {
                Circle()
                    .fill(isOnline ? Theme.success : Theme.danger)
                    .frame(width: 9, height: 9)
                Text(isOnline ? "Online" : "Offline")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(isOnline ? Theme.success : Theme.danger)
                Text(camera.model)
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
                Spacer()
                if camera.capabilities.doorbell {
                    Label("Doorbell", systemImage: "bell.fill")
                        .font(.caption)
                        .foregroundStyle(Theme.textSecondary)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
        }
        .background(Theme.cardBackground())
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    private func toggleLiveMute() {
        guard camera.capabilities.mic else { return }
        // No playback session here — live is WebRTC/VPIO and needs
        // .playAndRecord, which WHEPPlayer applies from isMuted.didSet.
        // Forcing .playback first is what silenced live audio + the mic.
        live.isMuted.toggle()
    }

    /// Talk needs the live receive audio audible to be a true two-way call.
    /// While a talk session is connecting/live, force the live player unmuted
    /// (remembering the user's prior mute state); when talk returns to idle,
    /// restore that state — re-activating playback if it was previously on.
    private func handleTalkAudio(_ talkState: TalkController.State) {
        guard camera.capabilities.mic else { return }
        switch talkState {
        case .connecting, .live:
            if muteBeforeTalk == nil {
                muteBeforeTalk = live.isMuted
            }
            if live.isMuted {
                live.isMuted = false
            }
        case .idle:
            guard let previous = muteBeforeTalk else { return }
            muteBeforeTalk = nil
            if previous {
                live.isMuted = true
            } else {
                // Talk deactivated the shared audio session on stop. Re-assert
                // the LIVE (.playAndRecord) session — NOT a playback one, which
                // would silence VPIO. A plain assignment is enough: Swift fires
                // didSet even when the value is unchanged, so WHEPPlayer
                // re-activates. Do NOT bounce it through `true` first — muting
                // now DEACTIVATES the session, so that churned it off/on during
                // talk teardown.
                live.isMuted = false
            }
        }
    }

    // MARK: - Control tabs (PTZ + two-way cameras)

    /// Segmented picker over the control cards so a PTZ+talk camera's controls
    /// are one tap away instead of a long scroll. Only used when usesControlTabs.
    private var controlTabs: some View {
        VStack(spacing: 16) {
            Picker("Controls", selection: $controlTab) {
                ForEach(ControlTab.allCases) { tab in
                    Text(tab.rawValue).tag(tab)
                }
            }
            .pickerStyle(.segmented)

            switch controlTab {
            case .ptz:
                ptzCard
            case .audio:
                talkCard
            case .camera:
                if showsControls {
                    controlsCard
                }
            }
        }
    }

    // MARK: - Controls card

    private var controlsCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            cardTitle("Controls", systemImage: "slider.horizontal.3")

            if !isOnline {
                Text("Camera is offline — controls are unavailable.")
                    .font(.caption)
                    .foregroundStyle(Theme.warning)
            }

            if hasDeviceSettings {
                if deviceLoading && device == nil {
                    HStack(spacing: 8) {
                        ProgressView().tint(Theme.accent)
                        Text("Reading device state…")
                            .font(.caption)
                            .foregroundStyle(Theme.textSecondary)
                    }
                } else if let deviceError, device == nil {
                    Text(deviceError)
                        .font(.caption)
                        .foregroundStyle(Theme.dangerSoft)
                }
            }

            // Night vision is now available on every camera, so the
            // Auto/Full-color/IR picker shows on all of them. The IR LED is
            // auto-managed by the backend, so the old separate IR Auto/On/Off
            // control is gone (it drove the same Dahua day/night table).
            nightVisionControl
            if camera.capabilities.whiteLight {
                spotlightControl
            }
            if camera.capabilities.siren {
                sirenControl
            }
            rebootControl
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.cardBackground())
        .disabled(!isOnline)
        .opacity(isOnline ? 1 : 0.55)
    }

    /// Night-vision picker (shown on every camera): Auto | Full-color | IR,
    /// writing `night_vision_mode` auto|color|bw through the device-settings
    /// PUT. This is the sole day/night control now that the IR LED is
    /// auto-managed by the backend.
    private var nightVisionControl: some View {
        VStack(alignment: .leading, spacing: 6) {
            controlLabel("Night vision", systemImage: "moon.stars.fill")
            Picker("Night vision", selection: $nightVisionMode) {
                ForEach(NightVisionMode.allCases) { mode in
                    Text(mode.label).tag(mode)
                }
            }
            .pickerStyle(.segmented)
            .disabled(device == nil)
            .onChange(of: nightVisionMode) { _, newMode in
                guard device?.nightVisionMode != newMode.rawValue else { return }
                Task { await applyNightVisionMode(newMode) }
            }
        }
    }

    private var spotlightControl: some View {
        VStack(alignment: .leading, spacing: 6) {
            controlLabel("Spotlight", systemImage: "lightbulb.max.fill")
            Picker("Spotlight", selection: $spotlightMode) {
                ForEach(SpotlightMode.allCases) { mode in
                    Text(mode.label).tag(mode)
                }
            }
            .pickerStyle(.segmented)
            .disabled(device == nil)
            .onChange(of: spotlightMode) { _, newMode in
                guard let current = device?.whiteLight?.mode, current != newMode.rawValue else { return }
                Task { await applySpotlight(mode: newMode) }
            }
        }
    }

    private var sirenControl: some View {
        VStack(alignment: .leading, spacing: 6) {
            controlLabel("Siren", systemImage: "speaker.wave.3.fill")
            HoldToConfirmButton(
                title: sirenBusy ? "Sounding siren…" : "Hold to sound siren (10 s)",
                systemImage: "light.beacon.max.fill",
                tint: Theme.danger
            ) {
                Task { await soundSiren() }
            }
            .disabled(sirenBusy)
        }
    }

    private var rebootControl: some View {
        VStack(alignment: .leading, spacing: 6) {
            controlLabel("Maintenance", systemImage: "wrench.and.screwdriver.fill")
            Button {
                showRebootConfirm = true
            } label: {
                HStack {
                    if rebootBusy {
                        ProgressView().tint(Theme.warning)
                    } else {
                        Image(systemName: "arrow.clockwise.circle.fill")
                    }
                    Text(rebootBusy ? "Rebooting…" : "Reboot camera")
                }
                .font(.callout.weight(.semibold))
                .foregroundStyle(Theme.warning)
                .frame(maxWidth: .infinity)
                .frame(height: 44)
                .background(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .fill(Theme.warning.opacity(0.12))
                )
            }
            .disabled(rebootBusy)
        }
    }

    // MARK: - PTZ card

    private var ptzCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            cardTitle("Pan / tilt / zoom", systemImage: "dpad.fill")

            if !isOnline {
                Text("Camera is offline — PTZ is unavailable.")
                    .font(.caption)
                    .foregroundStyle(Theme.warning)
            } else {
                Text("Tap an arrow to nudge one step. Hold a preset to save the current view, tap a saved preset to recall it, ✕ to clear.")
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
            }

            PTZControlsView(
                onStep: { direction in Task { await ptzStep(direction) } },
                onPresetGoto: { index in Task { await ptzPreset(.presetGoto, index: index) } },
                onPresetSet: { index in Task { await ptzPreset(.presetSet, index: index) } },
                onPresetClear: { index in Task { await ptzPreset(.presetClear, index: index) } },
                savedPresets: ptzSavedPresets,
                presetBusy: ptzPresetBusy,
                speed: $ptzSpeed,
                enabled: isOnline
            )
            .frame(maxWidth: .infinity)

            if let ptzFeedback {
                Label(ptzFeedback, systemImage: "checkmark.circle.fill")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(Theme.success)
                    .transition(.opacity)
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.cardBackground())
        .disabled(!isOnline)
        .opacity(isOnline ? 1 : 0.55)
    }

    // MARK: - Talk card

    private var talkCard: some View {
        VStack(spacing: 12) {
            cardTitle("Two-way talk", systemImage: "mic.fill")
                .frame(maxWidth: .infinity, alignment: .leading)

            // Every speaker camera uses the SAME simple WS talk pipeline
            // (mic → 8 kHz Int16 PCM → binary frames over the talk WebSocket).
            // The backend routes to the RTSP backchannel or CGI postAudio per
            // camera. The live receive connection stays strictly recvonly.
            PushToTalkButton(model: talk, state: \.state) {
                guard let url = session.api?.talkWebSocketURL(camera: camera.name) else {
                    controlError = "Talk connection URL is unavailable."
                    return
                }
                talk.start(url: url, protocols: session.api?.wsSubprotocols() ?? [])
            } onRelease: {
                talk.stop()
            }
            .disabled(!isOnline)
            .opacity(isOnline ? 1 : 0.55)

            Text(talkStatusText)
                .font(.caption)
                .foregroundStyle(talk.state == .live ? Theme.success : Theme.textSecondary)
        }
        .padding(14)
        .frame(maxWidth: .infinity)
        .background(Theme.cardBackground())
    }

    private var talkStatusText: String {
        if !isOnline { return "Camera is offline" }
        switch talk.state {
        case .idle: return "Hold the mic to talk through the camera speaker"
        case .connecting: return "Connecting…"
        case .live: return "Live — release to stop (2 min max)"
        }
    }

    // MARK: - AI detection setting

    // MARK: - Recent events

    private var recentEventsCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            cardTitle("Recent events", systemImage: "bell.badge.fill")

            if recentEvents.isEmpty {
                Text("No recent detections on this camera.")
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
            } else {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 10) {
                        ForEach(recentEvents) { event in
                            // Tap-through to the full event screen (clip
                            // playback + the Save/Share card).
                            NavigationLink {
                                EventDetailView(eventID: event.id)
                            } label: {
                                RecentEventTile(event: event, api: session.api)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.cardBackground())
    }

    // MARK: - Small view helpers

    private func cardTitle(_ title: String, systemImage: String) -> some View {
        Label(title, systemImage: systemImage)
            .font(.subheadline.weight(.semibold))
            .foregroundStyle(Theme.textPrimary)
    }

    private func controlLabel(_ title: String, systemImage: String) -> some View {
        Label(title, systemImage: systemImage)
            .font(.caption.weight(.semibold))
            .foregroundStyle(Theme.textSecondary)
    }

    // MARK: - Data loading

    private func initialLoad() async {
        async let events: Void = loadRecentEvents()
        async let settings: Void = loadDeviceSettings()
        _ = await (events, settings)
    }

    private func loadRecentEvents() async {
        guard let api = session.api else { return }
        do {
            let page = try await api.events(camera: camera.name, limit: 10)
            recentEvents = page.events
        } catch {
            session.handleAPIError(error)
        }
    }

    private func loadDeviceSettings() async {
        guard session.isAdmin, hasDeviceSettings, isOnline, let api = session.api else { return }
        deviceLoading = true
        defer { deviceLoading = false }
        do {
            let settings = try await api.cameraDeviceSettings(camera.name)
            device = settings
            deviceError = nil
            syncControls(from: settings)
        } catch {
            session.handleAPIError(error)
            deviceError = (error as? ApiError)?.message ?? error.localizedDescription
        }
    }

    private func syncControls(from settings: DeviceSettings) {
        if let raw = settings.nightVisionMode, let mode = NightVisionMode(rawValue: raw) {
            nightVisionMode = mode
        }
        if let light = settings.whiteLight,
           let mode = SpotlightMode(rawValue: light.mode) {
            spotlightMode = mode
        }
    }

    // MARK: - Control actions

    private func applyNightVisionMode(_ mode: NightVisionMode) async {
        guard let api = session.api else { return }
        do {
            let updated = try await api.updateCameraDeviceSettings(
                camera.name,
                patch: DeviceSettingsPatch(nightVisionMode: mode.rawValue)
            )
            device = updated
            syncControls(from: updated)
        } catch {
            session.handleAPIError(error)
            controlError = friendlyControlError(error, verb: "set the night-vision mode")
            // Roll the picker back to the device's actual state.
            if let raw = device?.nightVisionMode, let current = NightVisionMode(rawValue: raw) {
                nightVisionMode = current
            }
        }
    }

    // MARK: - PTZ actions

    /// One tap == one small step: a single self-contained POST with
    /// action:"step" + direction (no hold, no follow-up stop). The pad's speed
    /// slider sets the step magnitude (1–8).
    private func ptzStep(_ direction: PTZDirection) async {
        guard let api = session.api else { return }
        do {
            try await api.ptz(
                camera: camera.name, action: .step,
                direction: direction, speed: Int(ptzSpeed)
            )
        } catch {
            session.handleAPIError(error)
            controlError = friendlyPTZError(error, verb: "move the camera")
        }
    }

    private func ptzPreset(_ action: PTZAction, index: Int) async {
        guard let api = session.api else { return }
        ptzPresetBusy = index
        defer { ptzPresetBusy = nil }
        do {
            try await api.ptz(camera: camera.name, action: action, index: index)
            // Success — reconcile the saved-slot map + flash a confirmation.
            switch action {
            case .presetSet:
                ptzSavedPresets.insert(index)
                persistSavedPresets()
                showPTZFeedback("Saved preset \(index)")
            case .presetClear:
                ptzSavedPresets.remove(index)
                persistSavedPresets()
                showPTZFeedback("Cleared preset \(index)")
            case .presetGoto:
                showPTZFeedback("Recalled preset \(index)")
            default:
                break
            }
        } catch {
            session.handleAPIError(error)
            let verb: String
            switch action {
            case .presetSet: verb = "save preset \(index)"
            case .presetGoto: verb = "go to preset \(index)"
            case .presetClear: verb = "clear preset \(index)"
            default: verb = "run the preset"
            }
            controlError = friendlyPTZError(error, verb: verb)
        }
    }

    /// Flash a short PTZ confirmation caption, auto-clearing after ~2s. Guarded
    /// on the index so a newer action's caption isn't wiped by an older timer.
    private func showPTZFeedback(_ text: String) {
        withAnimation { ptzFeedback = text }
        Task {
            try? await Task.sleep(nanoseconds: 2_000_000_000)
            if ptzFeedback == text {
                withAnimation { ptzFeedback = nil }
            }
        }
    }

    // Saved-preset persistence (per camera). The dome exposes no preset list,
    // so we remember which slots the user has saved across launches.
    private var ptzPresetsDefaultsKey: String { "ptz_saved_presets_\(camera.name)" }

    private func loadSavedPresets() {
        let stored = UserDefaults.standard.array(forKey: ptzPresetsDefaultsKey) as? [Int] ?? []
        ptzSavedPresets = Set(stored.filter { (1 ... 3).contains($0) })
    }

    private func persistSavedPresets() {
        UserDefaults.standard.set(ptzSavedPresets.sorted(), forKey: ptzPresetsDefaultsKey)
    }

    /// PTZ error copy: 501 == the firmware/route doesn't support PTZ, 502 ==
    /// the camera's PTZ motor is busy with another command.
    private func friendlyPTZError(_ error: Error, verb: String) -> String {
        if let apiError = error as? ApiError {
            if apiError.status == 501 {
                return "This camera doesn't support PTZ."
            }
            if apiError.status == 502 {
                return "The camera is busy — try that again in a moment."
            }
        }
        return friendlyControlError(error, verb: verb)
    }

    private func applySpotlight(mode: SpotlightMode) async {
        guard let api = session.api else { return }
        do {
            try await api.setLight(camera: camera.name, mode: mode)
            var updated = device ?? DeviceSettings()
            // On/Off/Auto only — brightness isn't a knob on these turrets, so
            // preserve whatever the device last reported.
            updated.whiteLight = WhiteLightState(
                mode: mode.rawValue,
                brightness: updated.whiteLight?.brightness ?? 0
            )
            device = updated
        } catch {
            session.handleAPIError(error)
            controlError = friendlyControlError(error, verb: "set the spotlight")
            if let light = device?.whiteLight,
               let current = SpotlightMode(rawValue: light.mode) {
                spotlightMode = current
            }
        }
    }

    private func soundSiren() async {
        guard let api = session.api else { return }
        sirenBusy = true
        defer { sirenBusy = false }
        do {
            try await api.soundSiren(camera: camera.name, durationS: 10)
        } catch {
            session.handleAPIError(error)
            controlError = friendlyControlError(error, verb: "sound the siren")
        }
    }

    private func reboot() async {
        guard let api = session.api else { return }
        rebootBusy = true
        defer { rebootBusy = false }
        do {
            try await api.rebootCamera(camera.name)
            controlError = "Reboot sent — the camera will be back in about a minute."
        } catch {
            session.handleAPIError(error)
            controlError = friendlyControlError(error, verb: "reboot the camera")
        }
    }

    /// 501 == firmware rejected the CGI (contract); everything else verbatim.
    private func friendlyControlError(_ error: Error, verb: String) -> String {
        if let apiError = error as? ApiError {
            if apiError.status == 501 {
                return "The camera firmware doesn't support this command."
            }
            if apiError.status == 403 {
                return "Admin access is required to \(verb)."
            }
            return "Couldn't \(verb): \(apiError.message)"
        }
        return "Couldn't \(verb): \(error.localizedDescription)"
    }
}

// MARK: - RecentEventTile

/// One thumbnail in the recent-events strip: snapshot, label, relative time.
private struct RecentEventTile: View {
    let event: Event
    let api: APIClient?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ZStack {
                Rectangle().fill(Theme.bgDeep)
                if event.hasSnapshot, let api {
                    AsyncImage(url: api.eventSnapshotURL(id: event.id)) { phase in
                        if case .success(let image) = phase {
                            image.resizable().scaledToFill()
                        } else {
                            Image(systemName: "photo")
                                .foregroundStyle(Theme.textSecondary)
                        }
                    }
                } else {
                    Image(systemName: labelIcon)
                        .font(.title3)
                        .foregroundStyle(Theme.textSecondary)
                }
            }
            .frame(width: 132, height: 74)
            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .stroke(Theme.border, lineWidth: 1)
            )

            HStack(spacing: 4) {
                Text(event.label.replacingOccurrences(of: "_", with: " ").capitalized)
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(Theme.textPrimary)
                    .lineLimit(1)
                if event.count > 1 {
                    Text("×\(event.count)")
                        .font(.caption2)
                        .foregroundStyle(Theme.accent)
                }
            }
            Text(relativeTime)
                .font(.caption2)
                .foregroundStyle(Theme.textSecondary)
        }
        .frame(width: 132, alignment: .leading)
    }

    private var labelIcon: String {
        switch event.label {
        case "person": return "figure.walk"
        case "car", "truck", "bus": return "car.fill"
        case "dog", "cat": return "pawprint.fill"
        default: return "sparkles"
        }
    }

    private var relativeTime: String {
        let date = Date(timeIntervalSince1970: event.startTime)
        return date.formatted(.relative(presentation: .named))
    }
}

// MARK: - HoldToConfirmButton

/// A destructive action that requires a sustained press: a fill sweeps across
/// the button during the hold; releasing early cancels.
private struct HoldToConfirmButton: View {
    let title: String
    let systemImage: String
    let tint: Color
    let action: () -> Void

    @State private var progress: CGFloat = 0

    private let holdDuration: TimeInterval = 1.2

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(tint.opacity(0.12))
            GeometryReader { geo in
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(tint.opacity(0.35))
                    .frame(width: geo.size.width * progress)
            }
            HStack(spacing: 8) {
                Image(systemName: systemImage)
                Text(title)
            }
            .font(.callout.weight(.semibold))
            .foregroundStyle(tint)
        }
        .frame(height: 44)
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
        .contentShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
        .onLongPressGesture(minimumDuration: holdDuration, maximumDistance: 40) {
            withAnimation(.easeOut(duration: 0.2)) { progress = 0 }
            action()
        } onPressingChanged: { pressing in
            if pressing {
                withAnimation(.linear(duration: holdDuration)) { progress = 1 }
            } else {
                withAnimation(.easeOut(duration: 0.15)) { progress = 0 }
            }
        }
    }
}

// MARK: - PushToTalkButton

/// Hold-to-talk mic button. Press starts a talk session, release stops it. It's
/// generic over the controller (`state` is a key path to its
/// `TalkController.State`) so the affordance stays reusable; every speaker
/// camera drives the one `TalkController` WS pipeline and the backend routes
/// delivery (RTSP backchannel or CGI postAudio) per camera.
///
/// INTERNAL, not private: `SingleCameraView` shows the same affordance over
/// fullscreen live so an answered doorbell call can actually be answered. It
/// stays in this file rather than moving to its own because the Xcode project
/// lists sources explicitly — a new file is invisible to the build until
/// project.pbxproj is hand-edited, which is not worth it for one struct.
struct PushToTalkButton<Model: ObservableObject>: View {
    @ObservedObject var model: Model
    let state: KeyPath<Model, TalkController.State>
    /// Diameter in points. Declared BEFORE the two closures on purpose: Swift
    /// requires arguments in declaration order, so a `diameter:` after them
    /// could not be passed alongside the onPress/onRelease trailing closures.
    /// The controls screen uses the full size; the fullscreen live overlay sits
    /// this beside the mute pill over video, where 112 would cover the visitor
    /// being talked to.
    var diameter: CGFloat = 112
    let onPress: () -> Void
    let onRelease: () -> Void

    @State private var pressed = false

    private var talkState: TalkController.State { model[keyPath: state] }

    /// Scale the contents with the frame, so the compact form reads as a small
    /// button rather than a large one squeezed down.
    private var glyphSize: CGFloat { diameter * 0.30 }
    private var showsCaption: Bool { diameter >= 88 }

    var body: some View {
        ZStack {
            Circle()
                .fill(fillColor.opacity(talkState == .live ? 0.30 : 0.15))
                .frame(width: diameter, height: diameter)
            Circle()
                .stroke(fillColor, lineWidth: talkState == .live ? 3 : 1.5)
                .frame(width: diameter, height: diameter)
            VStack(spacing: 6) {
                Image(systemName: talkState == .live ? "mic.fill" : "mic")
                    .font(.system(size: glyphSize, weight: .medium))
                if showsCaption {
                    Text(buttonCaption)
                        .font(.caption2.weight(.semibold))
                }
            }
            .foregroundStyle(fillColor)
        }
        .scaleEffect(pressed ? 1.06 : 1)
        .animation(.spring(duration: 0.25), value: pressed)
        .animation(.easeInOut(duration: 0.2), value: talkState)
        .gesture(
            DragGesture(minimumDistance: 0)
                .onChanged { _ in
                    guard !pressed else { return }
                    pressed = true
                    onPress()
                }
                .onEnded { _ in
                    pressed = false
                    onRelease()
                }
        )
    }

    private var fillColor: Color {
        switch talkState {
        case .live: return Theme.success
        case .connecting: return Theme.warning
        case .idle: return Theme.accent
        }
    }

    private var buttonCaption: String {
        switch talkState {
        case .live: return "TALKING"
        case .connecting: return "…"
        case .idle: return "HOLD"
        }
    }
}

/// "Privacy Mode" panel for the camera-detail player card — the counterpart to
/// `PrivacyModeTileOverlay`. Replaces the player (and its retry-able failure
/// overlay) so an admin-chosen privacy state never reads as a camera fault.
private struct PrivacyModeDetailOverlay: View {
    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: "eye.slash.fill")
                .font(.title)
                .foregroundStyle(Color.white.opacity(0.85))
            Text("Privacy Mode")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(Color.white.opacity(0.9))
            Text("Not recording, detecting or streaming.")
                .font(.caption)
                .foregroundStyle(Color.white.opacity(0.6))
                .multilineTextAlignment(.center)
        }
        .padding(16)
        .accessibilityElement(children: .combine)
    }
}
