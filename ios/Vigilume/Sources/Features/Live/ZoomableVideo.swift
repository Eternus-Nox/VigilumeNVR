import SwiftUI

/// Wraps a full-bleed video surface (the WebRTC/HLS live layer or an AVPlayer
/// clip layer) with pinch-to-zoom + pan, shared by the full-screen live view
/// and the recorded-clip full-screen player.
///
/// - Pinch scales the content 1×…4× (live values clamped so it never overshoots).
/// - One-finger drag pans **only while zoomed**; the pan is clamped so the
///   scaled content edges can't be dragged inside the frame.
/// - Double-tap toggles between 1× (reset, re-centered) and 2×.
/// - A single tap is forwarded to `onSingleTap` so the host's existing gesture
///   (unmute on live, dismiss on the clip cover) keeps working. Because drag is
///   inert at 1×, that single-tap behavior is unchanged until the user zooms in.
///
/// The gestures live on the zoom layer itself; overlay chrome (close/mute
/// buttons, state overlay) sits above this view in its ZStack and keeps its own
/// hit testing, so controls are never blocked.
struct ZoomableVideo<Content: View>: View {
    var onSingleTap: () -> Void = {}
    @ViewBuilder var content: () -> Content

    private let minScale: CGFloat = 1
    private let maxScale: CGFloat = 4

    @State private var scale: CGFloat = 1
    @State private var offset: CGSize = .zero
    @GestureState private var pinch: CGFloat = 1
    @GestureState private var drag: CGSize = .zero
    @State private var containerSize: CGSize = .zero

    var body: some View {
        GeometryReader { geo in
            let liveScale = clampedScale(scale * pinch)
            content()
                .frame(width: geo.size.width, height: geo.size.height)
                .scaleEffect(liveScale)
                .offset(effectiveOffset(for: liveScale))
                .frame(width: geo.size.width, height: geo.size.height)
                .clipped()
                .contentShape(Rectangle())
                .gesture(magnification)
                .simultaneousGesture(panGesture)
                .onTapGesture(count: 2) { handleDoubleTap() }
                .onTapGesture(count: 1) { onSingleTap() }
                .onAppear { containerSize = geo.size }
                .onChange(of: geo.size) { _, newSize in
                    containerSize = newSize
                    clampOffset()
                }
        }
    }

    // MARK: Gestures

    private var magnification: some Gesture {
        MagnificationGesture()
            .updating($pinch) { value, state, _ in state = value }
            .onEnded { value in
                scale = clampedScale(scale * value)
                withAnimation(.easeOut(duration: 0.18)) { clampOffset() }
            }
    }

    private var panGesture: some Gesture {
        DragGesture()
            .updating($drag) { value, state, _ in
                guard scale > 1 else { return }
                state = value.translation
            }
            .onEnded { value in
                guard scale > 1 else { return }
                offset.width += value.translation.width
                offset.height += value.translation.height
                withAnimation(.easeOut(duration: 0.18)) { clampOffset() }
            }
    }

    private func handleDoubleTap() {
        withAnimation(.easeInOut(duration: 0.25)) {
            if scale > minScale {
                scale = minScale
                offset = .zero
            } else {
                scale = 2
            }
        }
    }

    // MARK: Math

    private func clampedScale(_ value: CGFloat) -> CGFloat {
        min(max(value, minScale), maxScale)
    }

    /// Committed offset + the in-flight drag, clamped so content stays covering.
    private func effectiveOffset(for liveScale: CGFloat) -> CGSize {
        let raw = CGSize(width: offset.width + drag.width,
                         height: offset.height + drag.height)
        let maxX = max(0, (liveScale - 1) * containerSize.width / 2)
        let maxY = max(0, (liveScale - 1) * containerSize.height / 2)
        return CGSize(
            width: min(max(raw.width, -maxX), maxX),
            height: min(max(raw.height, -maxY), maxY)
        )
    }

    private func clampOffset() {
        guard scale > 1 else { offset = .zero; return }
        let maxX = max(0, (scale - 1) * containerSize.width / 2)
        let maxY = max(0, (scale - 1) * containerSize.height / 2)
        offset.width = min(max(offset.width, -maxX), maxX)
        offset.height = min(max(offset.height, -maxY), maxY)
    }
}
