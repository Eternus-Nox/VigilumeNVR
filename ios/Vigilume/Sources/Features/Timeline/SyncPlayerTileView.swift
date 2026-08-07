import AVFoundation
import SwiftUI

/// One camera tile in the synced grid: an AVPlayerLayer render (NO native
/// controls — the shared transport bar is the only transport), a friendly-name
/// label, a remove (✕) button, and a "No footage at this time" placeholder
/// overlay when this camera's hour window is empty while the group keeps
/// playing. Mirrors the web `.sp-tile`.
struct SyncPlayerTileView: View {
    @ObservedObject var model: SyncCameraPlayerModel
    var onRemove: (() -> Void)?

    var body: some View {
        ZStack {
            PlayerLayerView(player: model.player, videoGravity: .resizeAspectFill)

            if !model.hasWindow {
                VStack(spacing: 6) {
                    Image(systemName: "video.slash")
                        .font(.title3)
                        .foregroundStyle(Theme.textSecondary)
                    Text("No footage at this time")
                        .font(.caption)
                        .foregroundStyle(Theme.textSecondary)
                        .multilineTextAlignment(.center)
                }
                .padding(8)
            }

            VStack {
                HStack(alignment: .top) {
                    Text(model.friendlyName.isEmpty ? model.camera : model.friendlyName)
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.white)
                        .lineLimit(1)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Capsule().fill(.black.opacity(0.55)))
                    Spacer()
                    if let onRemove {
                        Button(action: onRemove) {
                            Image(systemName: "xmark")
                                .font(.caption2.weight(.bold))
                                .foregroundStyle(.white)
                                .padding(6)
                                .background(Circle().fill(.black.opacity(0.55)))
                        }
                        .buttonStyle(.plain)
                        .accessibilityLabel("Remove \(model.friendlyName) from view")
                    }
                }
                Spacer()
            }
            .padding(8)
        }
        .aspectRatio(16 / 9, contentMode: .fit)
        .frame(maxWidth: .infinity)
        .background(Theme.bgDeep)
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .stroke(Theme.border, lineWidth: 1)
        )
    }
}
