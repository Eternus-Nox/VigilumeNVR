import SwiftUI

/// Stable, distinct hues per detection label — mirrors the web client's
/// LABEL_COLORS in frontend/src/components/TimelineBar.tsx. Anything unknown
/// falls back to slate. Shared by the events list and the timeline markers.
enum EventLabelStyle {
    private static let colors: [String: Color] = [
        "person": Color(hex: 0x38BDF8),
        "car": Color(hex: 0xA78BFA),
        "truck": Color(hex: 0xC084FC),
        "dog": Color(hex: 0xFBBF24),
        "cat": Color(hex: 0x34D399),
        "bicycle": Color(hex: 0xF472B6),
        "motorcycle": Color(hex: 0xFB923C),
    ]

    static func color(for label: String) -> Color {
        colors[label] ?? Color(hex: 0x94A3B8)
    }
}
