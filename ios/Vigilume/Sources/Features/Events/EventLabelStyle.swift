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

/// The one recognition worth showing beside an event.
///
/// A NAMED subject reads as a positive identification and a UNKNOWN one as a
/// caution, so they are visually distinct — but the distinction is carried by
/// the icon and the wording as much as the colour, because "unknown person at
/// the door" is exactly the row that must not depend on colour perception.
struct RecognitionBadge: View {
    let recognition: EventRecognition
    var compact: Bool = false

    private var isKnown: Bool { recognition.known && !recognition.name.isEmpty }

    private var icon: String {
        if recognition.isFace {
            return isKnown ? "person.crop.circle.badge.checkmark" : "person.crop.circle.badge.questionmark"
        }
        return isKnown ? "car.circle.fill" : "car.circle"
    }

    private var tint: Color { isKnown ? Theme.success : Theme.warning }

    var body: some View {
        // nil text means an unmatched face: the icon alone, in a round badge.
        // Spelling out "Unknown face" costs a badge's width on a thumbnail to
        // say nothing, and the wording survives on accessibilityLabel below.
        let text = recognition.printableText

        return HStack(spacing: text == nil ? 0 : 4) {
            Image(systemName: icon)
                .font(compact ? .caption2 : .caption)
            if let text {
                Text(text)
                    .font((compact ? Font.caption2 : Font.caption).weight(.semibold))
                    .lineLimit(1)
                    // Plates are strings of ambiguous glyphs; a monospaced face
                    // is what makes 0 vs O legible at this size.
                    .monospaced(!recognition.plate.isEmpty && !isKnown)
            }
        }
        .foregroundStyle(tint)
        .padding(.horizontal, text == nil ? 3 : 6)
        .padding(.vertical, text == nil ? 3 : 2)
        .background(Capsule().fill(tint.opacity(0.14)))
        .accessibilityLabel(
            isKnown
                ? "Recognized \(recognition.displayText)"
                : (recognition.isFace ? "Unrecognized face" : "Plate \(recognition.displayText)")
        )
    }
}
