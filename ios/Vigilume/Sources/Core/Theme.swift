import SwiftUI

/// Dark security-console palette, matching the web client's CSS
/// (frontend/src/*.css). The app is dark-only by design, like the PWA.
enum Theme {
    // Backgrounds
    static let bg = Color(hex: 0x0B1017)          // page background
    static let bgDeep = Color(hex: 0x0A0E14)      // recessed areas (tab bar, wells)
    static let surface = Color(hex: 0x131C2B)     // cards / list rows
    static let surfaceAlt = Color(hex: 0x0D1420)  // secondary cards
    static let elevated = Color(hex: 0x182333)    // sheets, popovers

    // Strokes
    static let border = Color(hex: 0x1E2A3D)
    static let borderStrong = Color(hex: 0x2C3F5E)

    // Text
    static let textPrimary = Color(hex: 0xDBE4F0)
    static let textSecondary = Color(hex: 0x8294AB)

    // Accents / status
    static let accent = Color(hex: 0x38BDF8)      // sky — primary interactive
    static let accentDeep = Color(hex: 0x0EA5E9)
    static let success = Color(hex: 0x34D399)     // online / ready
    static let warning = Color(hex: 0xFBBF24)     // degraded / unencrypted badge
    static let danger = Color(hex: 0xEF4444)      // offline / destructive
    static let dangerSoft = Color(hex: 0xF87171)

    /// Standard card background + border, used by feature views for tiles.
    static func cardBackground(cornerRadius: CGFloat = 12) -> some View {
        RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
            .fill(surface)
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .stroke(border, lineWidth: 1)
            )
    }
}

extension Color {
    /// `Color(hex: 0x38BDF8)`
    init(hex: UInt32, opacity: Double = 1) {
        self.init(
            .sRGB,
            red: Double((hex >> 16) & 0xFF) / 255,
            green: Double((hex >> 8) & 0xFF) / 255,
            blue: Double(hex & 0xFF) / 255,
            opacity: opacity
        )
    }
}
