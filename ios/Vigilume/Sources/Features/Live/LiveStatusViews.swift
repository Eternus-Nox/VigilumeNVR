import SwiftUI

/// Red-dot "LIVE" pill shown while a stream is actually rendering frames.
/// No latency claim — HLS live is seconds-class (docs/ios-design.md §2.3).
struct LiveBadge: View {
    var body: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(Theme.danger)
                .frame(width: 6, height: 6)
            Text("LIVE")
                .font(.caption2.weight(.bold))
                .foregroundStyle(.white)
        }
        .padding(.horizontal, 7)
        .padding(.vertical, 3)
        .background(Capsule().fill(Color.black.opacity(0.55)))
    }
}

/// "SD (compat)" pill + a small "HD" retry button, shown while the player is
/// on the substream fallback (docs/ios-design.md §2.1.1).
struct SDCompatBadge: View {
    let onRetryHD: () -> Void

    var body: some View {
        HStack(spacing: 0) {
            Text("SD (compat)")
                .font(.caption2.weight(.bold))
                .foregroundStyle(.white)
                .padding(.horizontal, 7)
                .padding(.vertical, 3)
            Rectangle()
                .fill(Color.white.opacity(0.25))
                .frame(width: 1, height: 12)
            Button(action: onRetryHD) {
                HStack(spacing: 3) {
                    Image(systemName: "arrow.clockwise")
                        .font(.system(size: 8, weight: .bold))
                    Text("HD")
                        .font(.caption2.weight(.bold))
                }
                .foregroundStyle(Theme.accent)
                .padding(.horizontal, 7)
                .padding(.vertical, 3)
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Retry HD stream")
        }
        .background(Capsule().fill(Color.black.opacity(0.55)))
    }
}

/// Centered status overlay for a live player: offline placeholder, connecting
/// spinner, reconnecting notice, or — when every source has failed — a real
/// error message with a retry button. Empty while playing.
struct PlayerStateOverlay: View {
    let state: LivePlayerModel.State
    let isOnline: Bool
    /// Real reason both streams failed (LivePlayerModel.failureText).
    var failureText: String? = nil
    /// Manual retry for the failure state (usually `retryPrimary`).
    var onRetry: (() -> Void)? = nil

    var body: some View {
        if !isOnline {
            VStack(spacing: 6) {
                Image(systemName: "wifi.slash")
                    .font(.title3)
                    .foregroundStyle(Theme.textSecondary)
                Text("Offline")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(Theme.textSecondary)
            }
        } else if let failureText, state != .playing {
            VStack(spacing: 8) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.title3)
                    .foregroundStyle(Theme.warning)
                Text("Live stream failed")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(Theme.textPrimary)
                Text(failureText)
                    .font(.caption2)
                    .foregroundStyle(Theme.textSecondary)
                    .multilineTextAlignment(.center)
                    .lineLimit(3)
                if let onRetry {
                    Button(action: onRetry) {
                        Label("Try again", systemImage: "arrow.clockwise")
                            .font(.caption.weight(.semibold))
                    }
                    .buttonStyle(.bordered)
                    .tint(Theme.accent)
                }
            }
            .padding(.horizontal, 24)
        } else {
            switch state {
            case .idle, .connecting:
                ProgressView()
                    .tint(Theme.textSecondary)
            case .retrying:
                VStack(spacing: 6) {
                    ProgressView()
                        .tint(Theme.textSecondary)
                    Text("Reconnecting…")
                        .font(.caption2)
                        .foregroundStyle(Theme.textSecondary)
                }
            case .playing:
                EmptyView()
            }
        }
    }
}

/// Instant first paint: the camera's most recent JPEG shown UNDER the video
/// layer while WHEP/HLS negotiates, faded out the moment a real decoded frame
/// lands.
///
/// WHY THIS EXISTS. Nothing here makes the stream connect any faster — it
/// removes the *perceived* wait, which is the whole startup. A live surface
/// otherwise shows flat `Theme.surfaceAlt` + a spinner for 0.3–2.2 s on the LAN
/// (dominated by the camera's next I-frame, which is camera-side and outside our
/// control) and far longer remote. `GET /api/cameras/{name}/snapshot.jpg` answers
/// from the detector's in-memory frame cache (`media.latest_jpg`) with no camera
/// round-trip, so it typically paints in ~50–150 ms on the LAN.
///
/// NEVER render this for a PRIVATE camera: that route 403s by design
/// (routers/cameras.py) and Privacy Mode must not be papered over with a stale
/// frame. Callers already branch on `camera.isPrivate` and show the privacy
/// overlay instead. Offline cameras are skipped too — the frame would be stale
/// with no live video coming to replace it.
struct LivePosterImage: View {
    let url: URL?
    /// True once real video is on screen — the poster's cue to get out of the way.
    let isPlaying: Bool
    /// Match the video layer's gravity so the swap doesn't visibly shift.
    var contentMode: ContentMode = .fill

    var body: some View {
        if let url, !isPlaying {
            // LAYOUT-NEUTRAL, deliberately. `Color.clear` accepts whatever size
            // the parent offers and an overlay can never push back on it, so the
            // decoded image cannot change the tile's dimensions. Rendering the
            // AsyncImage directly here made tiles RESIZE the moment the poster
            // decoded (a resizable image proposes its own ideal size into the
            // ZStack) — the grid visibly jumped mid-load.
            Color.clear
                .overlay {
                    AsyncImage(url: url) { phase in
                        if case .success(let image) = phase {
                            image
                                .resizable()
                                .aspectRatio(contentMode: contentMode)
                                .transition(.opacity)
                        }
                        // .empty / .failure: draw nothing and let the existing
                        // background + spinner show. A poster is best-effort; a
                        // camera with detection disabled and an unreachable CGI
                        // simply has no cached frame, and that must never
                        // surface as an error.
                    }
                }
                .clipped()
                // Purely decorative, and it sits UNDER the controls.
                .allowsHitTesting(false)
                .accessibilityHidden(true)
        }
    }
}
