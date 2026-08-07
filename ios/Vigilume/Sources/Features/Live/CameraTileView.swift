import SwiftUI

/// Shared plumbing for on-screen-only tile streaming (docs/ios-design.md §2).
enum LiveVisibility {
    /// Coordinate space CamerasView names on its ScrollView. A child's frame
    /// in this space is measured against the ScrollView's own bounds, so
    /// `minY` runs 0…viewportHeight while the child is on screen and goes
    /// negative / past the height once it scrolls out of view.
    static let coordinateSpace = "camerasLiveScroll"
}

/// Carries a single tile's on-screen boolean up to its own body via a
/// GeometryReader in the tile background. Reduce keeps `true` if any part
/// reports visible (there's only one reader per tile, so it's effectively a
/// pass-through).
private struct TileVisiblePreferenceKey: PreferenceKey {
    static var defaultValue = false
    static func reduce(value: inout Bool, nextValue: () -> Bool) {
        value = value || nextValue()
    }
}

/// One live dashboard tile: muted low-res substream (`<name>_sub`) via go2rtc
/// HLS. It holds a live AVPlayer **only while the tile is actually inside the
/// scroll viewport** — not merely instantiated by the LazyVStack. A tile that
/// scrolls off screen (or a backgrounded app) tears the player fully down
/// (stop → detach item, cancel watchdog/retry, drop the HLS session); it
/// rebuilds a fresh player when it scrolls back on. So the streaming set is
/// always exactly the visible set, never more.
///
/// The substream is the tile's `primary`; the full-res `main` stream is passed
/// as a `fallback` so a camera whose substream never produces a frame (the
/// AD410 doorbell has no usable `_sub`) doesn't sit on a black tile — after the
/// primary probe times out (~5 s), the model switches the tile to the main
/// stream. Auto-upgrade is disabled so the tile never flaps back to the broken
/// sub (see LivePlayerModel).
struct CameraTileView: View {
    let camera: Camera
    /// Preferred low-res substream (`<name>_sub`) HLS URL (WebRTC fallback).
    let streamURL: URL?
    /// Full-res main stream — used only if the substream can't render.
    let fallbackURL: URL?
    /// Sub-second WebRTC (WHEP) endpoint for the substream — tried first.
    let whepURL: URL?
    /// Cached-frame JPEG painted immediately, under the video, while the stream
    /// negotiates. See `LivePosterImage`.
    let posterURL: URL?
    let isOnline: Bool
    /// Height of the scroll viewport (from CamerasView) — the band a tile must
    /// intersect to count as visible. 0 until the parent has laid out.
    let viewportHeight: CGFloat
    let onTap: () -> Void

    @Environment(\.scenePhase) private var scenePhase
    // allowsAudio: false — a grid tile is permanently muted, so it must not even
    // NEGOTIATE audio. Muting alone only disables the track; the connection would
    // still carry audio, WebRTC would open its audio unit, and our
    // .playAndRecord session means that unit takes the MIC — so a grid of muted
    // tiles lit the mic indicator with nothing listening.
    @StateObject private var model = LiveController(autoUpgrade: false, allowsAudio: false)
    /// Driven by the background GeometryReader: is this tile on screen right now?
    @State private var isVisible = false

    var body: some View {
        ZStack {
            Theme.surfaceAlt

            if camera.isPrivate {
                PrivacyModeTileOverlay()
            } else {
                // UNDER the video: paints in ~50-150 ms so the tile is never a
                // grey box while WHEP negotiates. Skipped when offline (a stale
                // frame with no live video coming to replace it is a lie).
                if isOnline {
                    LivePosterImage(
                        url: posterURL,
                        isPlaying: model.state == .playing,
                        contentMode: .fill
                    )
                }

                LiveVideoLayer(controller: model, videoGravity: .resizeAspectFill)

                PlayerStateOverlay(state: model.state, isOnline: isOnline)
            }

            VStack {
                // Top-right only. The "AI" pill that used to sit top-left is
                // gone: on-camera AI firing is detector plumbing, not something
                // a person watching a camera grid needs on every tile.
                HStack(spacing: 6) {
                    Spacer()
                    if isOnline, !camera.isPrivate, model.state == .playing {
                        LiveBadge()
                    }
                }
                Spacer()
                nameBar
            }
            .padding(6)
        }
        .aspectRatio(16.0 / 9.0, contentMode: .fit)
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .stroke(Theme.border, lineWidth: 1)
        )
        .contentShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .onTapGesture(perform: onTap)
        // Visibility probe: reports whether this tile intersects the viewport.
        .background(
            GeometryReader { proxy in
                Color.clear.preference(
                    key: TileVisiblePreferenceKey.self,
                    value: isProxyVisible(proxy)
                )
            }
        )
        // WRITE-ONLY. This closure is @Sendable on iOS 18+, where capturing
        // view state is unsupported — the previous version read `isVisible`
        // inside it to early-return, which is exactly the pattern that stops
        // the write from landing. Store the raw report and let `.onChange`
        // below react; SwiftUI already coalesces equal values.
        // WRITE-ONLY, deliberately. On iOS 18+ this action is @Sendable, where
        // capturing view state is unsupported — the earlier version read
        // `isVisible` inside it to early-return. That happened not to be the
        // cause of any observed bug (the grid recovers on its own, so the
        // probe does fire), but it is the kind of thing that breaks silently on
        // an OS update, and SwiftUI already coalesces equal preference values.
        .onPreferenceChange(TileVisiblePreferenceKey.self) { visible in
            isVisible = visible
        }
        .onChange(of: isVisible) { _, _ in syncPlayback() }
        // onDisappear covers the case where the LazyVStack recycles the row
        // entirely (visibility preference may never fire a final `false`).
        .onDisappear { model.stop() }
        .onChange(of: camera.isPrivate) { _, _ in syncPlayback() }
        .onChange(of: isOnline) { _, _ in syncPlayback() }
        .onChange(of: streamURL) { _, _ in syncPlayback() }
        .onChange(of: fallbackURL) { _, _ in syncPlayback() }
        .onChange(of: whepURL) { _, _ in syncPlayback() }
        .onChange(of: scenePhase) { _, _ in syncPlayback() }
    }

    private var nameBar: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(isOnline ? Theme.success : Theme.danger)
                .frame(width: 7, height: 7)
            Text(camera.friendlyName)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.white)
                .lineLimit(1)
            Spacer(minLength: 0)
            if camera.capabilities.doorbell {
                Image(systemName: "bell.fill")
                    .font(.caption2)
                    .foregroundStyle(Color.white.opacity(0.7))
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 5)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(Color.black.opacity(0.45))
        )
    }

    /// A tile is visible when any part of its frame lies inside the viewport
    /// band (0…viewportHeight) in the ScrollView's coordinate space.
    private func isProxyVisible(_ proxy: GeometryProxy) -> Bool {
        guard viewportHeight > 0 else { return false }
        let frame = proxy.frame(in: .named(LiveVisibility.coordinateSpace))
        return frame.maxY > 0 && frame.minY < viewportHeight
    }

    /// Single source of truth for whether this tile should be streaming right
    /// now: on screen, online, foreground, and with a URL. Anything else tears
    /// the player fully down (grid tiles are cheap to rebuild — no `suspend`).
    private func syncPlayback() {
        // `isPrivate` first: the backend already refuses to serve a private
        // camera's stream, so attempting playback would only produce a
        // connection-failure overlay that reads as "broken" instead of
        // "deliberately off". Bailing here is cosmetic, not the enforcement —
        // privacy is enforced server-side and never trusts this client.
        guard !camera.isPrivate, scenePhase == .active, isVisible, isOnline,
              let streamURL
        else {
            model.stop()
            return
        }
        // WHEP sub first (sub-second); HLS sub as fallback; the main HLS stream
        // rescues tiles with no usable substream.
        // whepHigh: nil — a grid tile stays on the substream for good. Twelve
        // full-res WebRTC sessions is exactly the bandwidth/battery cliff the
        // sub rung exists to avoid.
        model.play(whepLow: whepURL, whepHigh: nil, primary: streamURL, fallback: fallbackURL)
    }
}

/// Full-tile "Privacy Mode" treatment for a camera in Software Privacy Mode.
///
/// Replaces the video layer outright rather than dimming it — there is no
/// footage to dim: the backend serves no stream, records nothing and detects
/// nothing for this camera. The wording says capture is OFF (not "unavailable")
/// so the state reads as deliberate rather than as a fault.
private struct PrivacyModeTileOverlay: View {
    var body: some View {
        ZStack {
            Theme.surfaceAlt
            VStack(spacing: 6) {
                Image(systemName: "eye.slash.fill")
                    .font(.title2)
                    .foregroundStyle(Color.white.opacity(0.85))
                Text("Privacy Mode")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(Color.white.opacity(0.85))
                Text("Not recording")
                    .font(.caption2)
                    .foregroundStyle(Color.white.opacity(0.55))
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Privacy Mode. This camera is not recording.")
    }
}
