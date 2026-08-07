import Foundation

/// Shared time math for the Timeline: local-day boundaries, hour-aligned VOD
/// windows, and the wall-clock <-> media-time mapping through a camera's real
/// segments (so coverage gaps map correctly). Mirrors the web client's
/// frontend/src/lib/timelineTime.ts — the recordings API uses the server's
/// LOCAL day, which we treat as the device's local day (same LAN/timezone).
enum TimelineTime {
    static let day: Double = 86_400
    static let hour: Double = 3_600
    /// Backend cap on GET /api/recordings/{cam}/export.mp4 (30 min).
    static let maxExportSeconds: Double = 30 * 60

    static func clamp(_ value: Double, _ lower: Double, _ upper: Double) -> Double {
        min(upper, max(lower, value))
    }

    // MARK: Local-day helpers

    /// Epoch seconds of local midnight for the given day.
    static func dayStart(of date: Date) -> Double {
        Calendar.current.startOfDay(for: date).timeIntervalSince1970
    }

    /// "YYYY-MM-DD" for the recordings index route (local day).
    static func dateString(for date: Date) -> String {
        let c = Calendar.current.dateComponents([.year, .month, .day], from: date)
        return String(format: "%04d-%02d-%02d", c.year ?? 0, c.month ?? 0, c.day ?? 0)
    }

    static func shift(_ date: Date, byDays days: Int) -> Date {
        Calendar.current.date(byAdding: .day, value: days, to: date) ?? date
    }

    static func isToday(_ date: Date) -> Bool {
        Calendar.current.isDateInToday(date) || date > Date()
    }

    // MARK: Hour window (player VOD window, aligned to recorder hour dirs)

    /// Hour window (start,end) containing wall-clock `t`, aligned to the local
    /// day so it matches the recorder's hour directories.
    static func hourWindow(containing t: Double, dayStart: Double, dayEnd: Double) -> (start: Double, end: Double) {
        let start = dayStart + (Double(Int((t - dayStart) / hour)) * hour)
        return (start, min(start + hour, dayEnd))
    }

    // MARK: Wall-clock <-> media-time mapping through real segments

    static func segments(
        _ segments: [RecordingSegment], inWindow window: (start: Double, end: Double)
    ) -> [RecordingSegment] {
        segments.filter { $0.start + $0.duration > window.start && $0.start < window.end }
    }

    /// Media (playlist) time for a wall-clock instant; handles coverage gaps.
    static func mediaTime(
        forWall t: Double,
        segments segs: [RecordingSegment],
        window: (start: Double, end: Double)
    ) -> Double {
        var acc: Double = 0
        for seg in self.segments(segs, inWindow: window) {
            let end = seg.start + seg.duration
            if t >= end {
                acc += seg.duration
                continue
            }
            if t <= seg.start { return acc }   // t sits in a gap before this segment
            return acc + (t - seg.start)
        }
        return acc
    }

    /// Inverse: wall-clock for a media-time offset (drives the playhead).
    static func wallTime(
        forMedia m: Double,
        segments segs: [RecordingSegment],
        window: (start: Double, end: Double)
    ) -> Double {
        var acc: Double = 0
        let inWindow = segments(segs, inWindow: window)
        for seg in inWindow {
            if m < acc + seg.duration { return seg.start + (m - acc) }
            acc += seg.duration
        }
        if let last = inWindow.last { return last.start + last.duration }
        return window.start
    }

    // MARK: Formatting

    static func clockLabel(_ epoch: Double, seconds: Bool = true) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = seconds ? "h:mm:ss a" : "h:mm a"
        return formatter.string(from: Date(timeIntervalSince1970: epoch))
    }

    /// "1h 05m" / "12m 30s" / "45s" for a span in seconds.
    static func spanLabel(_ span: Double) -> String {
        let s = Int(span.rounded())
        if s >= 3600 { return String(format: "%dh %02dm", s / 3600, (s % 3600) / 60) }
        if s >= 60 { return String(format: "%dm %02ds", s / 60, s % 60) }
        return "\(s)s"
    }
}
