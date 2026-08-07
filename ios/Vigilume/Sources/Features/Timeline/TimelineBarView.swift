import SwiftUI

/// The Timeline's single native scrub bar (like the web's unified lane bar):
/// renders, over the window [viewStart, viewEnd] (epoch seconds),
/// - union RECORDED coverage across all selected cameras (overlap shades
///   heavier where more cameras recorded),
/// - hour/minute tick labels,
/// - a draggable PLAYHEAD (tap/drag anywhere seeks),
/// - in range mode, a drag selects an export span (capped at 30 min).
struct TimelineBarView: View {
    let viewStart: Double
    let viewEnd: Double
    /// One ranges array per selected camera (union coverage with heat).
    let coverage: [[RecordingRange]]
    /// Event start times (epoch s) drawn as small inert marks, purely so you can
    /// SEE where events are and scrub to them. Painted into the Canvas rather
    /// than added as views, so they have no hit target and can never be tapped
    /// or steal a scrub — reviewing an event is the Events tab's job.
    let eventTimes: [Double]
    let playhead: Double

    let rangeMode: Bool
    @Binding var range: ClosedRange<Double>?

    let onScrubStart: () -> Void
    let onScrub: (Double) -> Void
    let onScrubEnd: (Double) -> Void

    @State private var dragAnchor: Double?
    @State private var isDragging = false

    private var span: Double { max(1, viewEnd - viewStart) }

    var body: some View {
        GeometryReader { geo in
            let width = geo.size.width
            ZStack(alignment: .topLeading) {
                canvas(width: width, height: geo.size.height)
                playheadKnob(width: width, height: geo.size.height)
            }
            .contentShape(Rectangle())
            .gesture(barGesture(width: width))
        }
        .frame(height: 76)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Theme.bgDeep)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .stroke(Theme.border, lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
    }

    // MARK: Coordinate mapping

    private func x(for t: Double, width: CGFloat) -> CGFloat {
        CGFloat((t - viewStart) / span) * width
    }

    private func time(at x: CGFloat, width: CGFloat) -> Double {
        viewStart + Double(min(max(0, x), width) / max(width, 1)) * span
    }

    // MARK: Canvas layers

    private func canvas(width: CGFloat, height: CGFloat) -> some View {
        Canvas { context, size in
            let trackTop: CGFloat = 6
            let trackBottom = size.height - 18
            let trackHeight = trackBottom - trackTop

            // Union coverage: each camera's ranges at low opacity — overlaps
            // accumulate into a heat shade.
            let layerOpacity = 0.16 + 0.24 / Double(max(1, coverage.count))
            for ranges in coverage {
                for r in ranges where r.end > viewStart && r.start < viewEnd {
                    let x0 = x(for: max(r.start, viewStart), width: size.width)
                    let x1 = x(for: min(r.end, viewEnd), width: size.width)
                    let rect = CGRect(x: x0, y: trackTop, width: max(1, x1 - x0), height: trackHeight)
                    context.fill(Path(rect), with: .color(Theme.accent.opacity(layerOpacity)))
                }
            }

            // Event locations: small inert marks sitting just above the clock
            // ticks. Painted (not views) so there is nothing to tap — they only
            // tell you WHERE to scrub.
            for t in eventTimes where t >= viewStart && t <= viewEnd {
                let ex = x(for: t, width: size.width)
                let markHeight: CGFloat = 10
                let rect = CGRect(
                    x: ex - 1, y: trackBottom - markHeight, width: 2, height: markHeight
                )
                context.fill(
                    Path(roundedRect: rect, cornerRadius: 1),
                    with: .color(Theme.warning.opacity(0.9))
                )
            }

            // Tick marks + clock labels.
            let tickStep = tickInterval()
            var tick = (viewStart / tickStep).rounded(.up) * tickStep
            while tick <= viewEnd {
                let tx = x(for: tick, width: size.width)
                var line = Path()
                line.move(to: CGPoint(x: tx, y: trackBottom - 6))
                line.addLine(to: CGPoint(x: tx, y: trackBottom))
                context.stroke(line, with: .color(Theme.borderStrong), lineWidth: 1)

                let label = Text(tickLabel(tick))
                    .font(.system(size: 9))
                    .foregroundColor(Theme.textSecondary)
                context.draw(label, at: CGPoint(x: tx, y: trackBottom + 9), anchor: .center)
                tick += tickStep
            }

            // Range-export selection overlay.
            if let range {
                let x0 = x(for: max(range.lowerBound, viewStart), width: size.width)
                let x1 = x(for: min(range.upperBound, viewEnd), width: size.width)
                if x1 > x0 {
                    let rect = CGRect(x: x0, y: trackTop, width: x1 - x0, height: trackHeight)
                    context.fill(Path(rect), with: .color(Theme.warning.opacity(0.25)))
                    for edge in [x0, x1] {
                        var line = Path()
                        line.move(to: CGPoint(x: edge, y: trackTop))
                        line.addLine(to: CGPoint(x: edge, y: trackBottom))
                        context.stroke(line, with: .color(Theme.warning), lineWidth: 2)
                    }
                }
            }
        }
    }

    /// Tick spacing tuned to the visible span (day -> 3 h, hour -> 10 min).
    private func tickInterval() -> Double {
        if span > 6 * TimelineTime.hour { return 3 * TimelineTime.hour }
        if span > TimelineTime.hour { return TimelineTime.hour }
        return 10 * 60
    }

    // Two shared formatters, built once.
    //
    // DateFormatter construction is among the most expensive routine operations
    // in Foundation — it builds an ICU formatter and consults the locale. This
    // was allocating a fresh one PER TICK LABEL, and the Canvas redraws on every
    // TimelineView body pass, which during a scrub is every touch sample: on the
    // order of a thousand ICU formatter constructions per second of dragging, to
    // render six or eight short strings.
    //
    // Safe to share: DateFormatter is thread-safe for reading on Apple platforms
    // since macOS 10.9/iOS 7, and these are never mutated after construction —
    // which is why the format is chosen by PICKING a formatter rather than by
    // assigning .dateFormat on a shared one.
    private static let hourFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "h a"
        return f
    }()

    private static let minuteFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "h:mm"
        return f
    }()

    private func tickLabel(_ t: Double) -> String {
        let formatter = span > 6 * TimelineTime.hour ? Self.hourFormatter : Self.minuteFormatter
        return formatter.string(from: Date(timeIntervalSince1970: t))
    }

    // MARK: Playhead

    private func playheadKnob(width: CGFloat, height: CGFloat) -> some View {
        let px = min(max(0, x(for: playhead, width: width)), width)
        return ZStack(alignment: .top) {
            Rectangle()
                .fill(Color.white)
                .frame(width: 2, height: height - 16)
            Circle()
                .fill(Color.white)
                .frame(width: 11, height: 11)
                .shadow(color: .black.opacity(0.6), radius: 2)
        }
        .position(x: px, y: (height - 16) / 2 + 3)
        .allowsHitTesting(false)
    }

    // MARK: Gestures (scrub, or range select in range mode)

    private func barGesture(width: CGFloat) -> some Gesture {
        DragGesture(minimumDistance: 0)
            .onChanged { value in
                let t = time(at: value.location.x, width: width)
                if rangeMode {
                    let anchor = dragAnchor ?? time(at: value.startLocation.x, width: width)
                    if dragAnchor == nil { dragAnchor = anchor }
                    var lo = min(anchor, t)
                    var hi = max(anchor, t)
                    // Clamp the span to the backend export cap, growing away
                    // from the anchor.
                    if hi - lo > TimelineTime.maxExportSeconds {
                        if t >= anchor { hi = lo + TimelineTime.maxExportSeconds }
                        else { lo = hi - TimelineTime.maxExportSeconds }
                    }
                    range = lo ... hi
                } else {
                    if !isDragging {
                        isDragging = true
                        onScrubStart()
                    }
                    onScrub(t)
                }
            }
            .onEnded { value in
                let t = time(at: value.location.x, width: width)
                if rangeMode {
                    dragAnchor = nil
                    if let r = range, r.upperBound - r.lowerBound < 1 {
                        range = nil   // a bare tap isn't a selection
                    }
                } else {
                    isDragging = false
                    onScrubEnd(t)
                }
            }
    }
}
