import SwiftUI

/// Settings › Excluded objects: the learned false-detection suppressions
/// (reject-to-suppress). Each row shows the suppression's thumbnail, its label,
/// the camera's friendly name, and when it was learned. Swipe to delete forgets
/// a suppression. Reachable from `SettingsHomeView` (admin-only), so it lives
/// inside that view's existing NavigationStack — no NavigationStack of its own.
struct SuppressionsView: View {
    @EnvironmentObject private var session: SessionModel

    @State private var suppressions: [Suppression] = []
    @State private var cameras: [Camera] = []
    @State private var loading = true
    @State private var errorMessage: String?

    var body: some View {
        Group {
            if let errorMessage, suppressions.isEmpty {
                ContentUnavailableView(
                    "Couldn't load exclusions",
                    systemImage: "eye.slash",
                    description: Text(errorMessage)
                )
            } else if suppressions.isEmpty && !loading {
                ContentUnavailableView(
                    "No exclusions",
                    systemImage: "eye.slash",
                    description: Text("Reject an event to stop alerting on detections like it.")
                )
            } else {
                List {
                    ForEach(suppressions) { suppression in
                        row(suppression)
                            .listRowBackground(Theme.surface)
                    }
                    .onDelete(perform: delete)
                }
                .listStyle(.plain)
                .scrollContentBackground(.hidden)
                .overlay {
                    if loading && suppressions.isEmpty {
                        ProgressView().tint(Theme.accent)
                    }
                }
            }
        }
        .background(Theme.bg)
        .navigationTitle("Excluded objects")
        .navigationBarTitleDisplayMode(.inline)
        .task { await load() }
        .refreshable { await load() }
    }

    // MARK: Row

    private func row(_ suppression: Suppression) -> some View {
        HStack(spacing: 12) {
            thumbnail(for: suppression)
            VStack(alignment: .leading, spacing: 3) {
                Text(suppression.label.capitalized)
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(Theme.textPrimary)
                Text(friendlyName(for: suppression.camera))
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
                Text(Date(timeIntervalSince1970: suppression.createdAt)
                        .formatted(date: .abbreviated, time: .shortened))
                    .font(.caption2)
                    .foregroundStyle(Theme.textSecondary)
            }
            Spacer(minLength: 0)
        }
        .padding(.vertical, 4)
    }

    /// Suppression thumbnail (106×60, rounded, bordered). A missing thumb (404)
    /// or in-flight load shows the photo-glyph placeholder — mirrors the Events
    /// list thumbnail.
    private func thumbnail(for suppression: Suppression) -> some View {
        AsyncImage(url: session.api?.suppressionThumbURL(id: suppression.id)) { image in
            image.resizable().aspectRatio(contentMode: .fill)
        } placeholder: {
            thumbPlaceholder
        }
        .frame(width: 106, height: 60)
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(Theme.border, lineWidth: 1)
        )
    }

    private var thumbPlaceholder: some View {
        Rectangle()
            .fill(Theme.surfaceAlt)
            .overlay(
                Image(systemName: "photo")
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
            )
    }

    private func friendlyName(for camera: String) -> String {
        if let match = cameras.first(where: { $0.name == camera }),
           !match.friendlyName.isEmpty {
            return match.friendlyName
        }
        return camera.replacingOccurrences(of: "_", with: " ").capitalized
    }

    // MARK: Loading

    private func load() async {
        guard let api = session.api else { return }
        if cameras.isEmpty {
            cameras = (try? await api.cameras()) ?? []
        }
        loading = suppressions.isEmpty
        do {
            suppressions = try await api.suppressions()
            errorMessage = nil
        } catch {
            session.handleAPIError(error)
            errorMessage = (error as? ApiError)?.message ?? error.localizedDescription
        }
        loading = false
    }

    // MARK: Delete

    private func delete(at offsets: IndexSet) {
        guard let api = session.api else { return }
        let targets = offsets.map { suppressions[$0] }
        Task {
            for target in targets {
                do {
                    try await api.deleteSuppression(id: target.id)
                    suppressions.removeAll { $0.id == target.id }
                } catch {
                    session.handleAPIError(error)
                    errorMessage = (error as? ApiError)?.message ?? error.localizedDescription
                }
            }
        }
    }
}
