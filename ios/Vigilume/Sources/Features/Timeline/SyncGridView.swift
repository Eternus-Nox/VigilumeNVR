import SwiftUI

/// Responsive grid of synchronized camera tiles for the on-view cameras.
/// One camera -> a single full-width 16:9 tile; two or more -> a 2-column grid
/// of 16:9 tiles. Mirrors the web SyncPlaybackGrid (cols = count == 1 ? 1 : 2).
struct SyncGridView: View {
    let models: [SyncCameraPlayerModel]
    var onRemove: ((String) -> Void)?

    private var columns: [GridItem] {
        let count = models.count <= 1 ? 1 : 2
        return Array(repeating: GridItem(.flexible(), spacing: 10), count: count)
    }

    var body: some View {
        if models.isEmpty {
            emptyTile
        } else {
            LazyVGrid(columns: columns, spacing: 10) {
                ForEach(models) { model in
                    SyncPlayerTileView(
                        model: model,
                        onRemove: onRemove.map { remove in { remove(model.camera) } }
                    )
                }
            }
            .padding(.horizontal, 16)
        }
    }

    private var emptyTile: some View {
        ZStack {
            VStack(spacing: 6) {
                Image(systemName: "rectangle.on.rectangle.slash")
                    .font(.title3)
                    .foregroundStyle(Theme.textSecondary)
                Text("No cameras on view")
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
            }
        }
        .frame(maxWidth: .infinity)
        .aspectRatio(16 / 9, contentMode: .fit)
        .background(Theme.bgDeep)
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .stroke(Theme.border, lineWidth: 1)
        )
        .padding(.horizontal, 16)
    }
}
