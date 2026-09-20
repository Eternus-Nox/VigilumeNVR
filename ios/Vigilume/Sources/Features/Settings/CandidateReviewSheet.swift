import SwiftUI

/// One unmatched sighting, shown against the WHOLE FRAME it came from, with
/// "add this to…" underneath.
///
/// WHY THE FRAME AND NOT THE CROP
/// ==============================
/// The grid behind this sheet shows what the recognizer sees: a 112 pt crop,
/// aligned to the canonical geometry, background gone. That is the correct
/// input for an embedding and a poor basis for a human decision — it cannot
/// say who else was in the shot, what the person was doing, or, on a doorstep
/// with three people, which of them this even is.
///
/// Enrolling the wrong face is the one mistake on this screen that lasts: from
/// then on the gallery matches a stranger as you, silently, with nothing
/// pointing back at the moment it went wrong. So the decision is made against
/// the scene, and the crop sits beside it as "this is the part that gets
/// enrolled".
///
/// `frameBox` rings the subject. It is drawn from a GeometryReader over the
/// image's own frame rather than over the sheet, because the image is letter-
/// boxed by `.scaledToFit` and a rectangle in sheet coordinates would sit
/// confidently in the wrong place. When the box is missing (a sighting from
/// before it was captured) nothing is drawn — no ring beats a wrong ring.
///
/// ENROLLING IS NOT TRAINING. It appends this shot's embedding to a profile's
/// gallery; nothing is fitted and removing the sample later undoes it
/// completely. That is why this is one confirming tap and not a warning.
struct CandidateReviewSheet: View {
    let candidate: RecognitionCandidate
    /// Already filtered to the kind that matches this candidate — a face
    /// cannot join a vehicle.
    let profiles: [RecognitionProfile]
    var onEnroll: (RecognitionProfile) async -> Void
    var onDelete: () async -> Void

    @EnvironmentObject private var session: SessionModel
    @Environment(\.dismiss) private var dismiss

    @State private var working = false

    private var isFace: Bool { candidate.kind == "face" }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    frame
                    crop
                    facts
                    addTo
                }
                .padding(16)
            }
            .background(Theme.bg)
            .navigationTitle(isFace ? "Who is this?" : "Which vehicle?")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Close") { dismiss() }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button(role: .destructive) {
                        Task {
                            working = true
                            await onDelete()
                            working = false
                            dismiss()
                        }
                    } label: {
                        Image(systemName: "trash")
                    }
                    .disabled(working)
                }
            }
        }
    }

    // MARK: Frame

    @ViewBuilder
    private var frame: some View {
        if candidate.hasFrame, let url = session.api?.recognitionCandidateFrameURL(id: candidate.id) {
            AsyncImage(url: url) { phase in
                switch phase {
                case .success(let image):
                    image
                        .resizable()
                        .scaledToFit()
                        .overlay { boxOverlay }
                case .failure:
                    unavailable("The full frame could not be loaded.")
                default:
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Theme.bgDeep)
                        .frame(height: 200)
                        .overlay { ProgressView().tint(Theme.accent) }
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 12))
        } else {
            unavailable(
                "The event this was seen in has already been deleted, so the full frame is gone. The crop below is all that is left of it."
            )
        }
    }

    /// The ring, in the IMAGE's coordinates.
    ///
    /// GeometryReader here reads the frame of the image view itself, which
    /// `.scaledToFit` has already sized to the aspect-correct box — so the
    /// normalized rectangle maps onto it directly. Reading the sheet's
    /// geometry instead would land the ring in the letterboxing.
    @ViewBuilder
    private var boxOverlay: some View {
        if let box = candidate.frameBox, box.count == 4 {
            GeometryReader { geo in
                let x0 = min(box[0], box[2]), x1 = max(box[0], box[2])
                let y0 = min(box[1], box[3]), y1 = max(box[1], box[3])
                RoundedRectangle(cornerRadius: 3)
                    .strokeBorder(Theme.accent, lineWidth: 2)
                    .frame(
                        width: max(8, CGFloat(x1 - x0) * geo.size.width),
                        height: max(8, CGFloat(y1 - y0) * geo.size.height)
                    )
                    .position(
                        x: CGFloat((x0 + x1) / 2) * geo.size.width,
                        y: CGFloat((y0 + y1) / 2) * geo.size.height
                    )
            }
        }
    }

    private func unavailable(_ message: String) -> some View {
        Text(message)
            .font(.footnote)
            .foregroundStyle(Theme.textSecondary)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(14)
            .background(RoundedRectangle(cornerRadius: 12).fill(Theme.surface))
    }

    // MARK: Crop

    private var crop: some View {
        HStack(spacing: 12) {
            Group {
                if candidate.hasImage,
                   let url = session.api?.recognitionCandidateImageURL(id: candidate.id) {
                    AsyncImage(url: url) { phase in
                        switch phase {
                        case .success(let image):
                            image.resizable().scaledToFill()
                        case .failure:
                            Rectangle().fill(Theme.bgDeep).overlay {
                                Image(systemName: "photo").foregroundStyle(Theme.textSecondary)
                            }
                        default:
                            Rectangle().fill(Theme.bgDeep)
                        }
                    }
                } else {
                    Rectangle().fill(Theme.bgDeep).overlay {
                        Image(systemName: "photo").foregroundStyle(Theme.textSecondary)
                    }
                }
            }
            .frame(width: 84, height: 84)
            .clipShape(RoundedRectangle(cornerRadius: 10))

            VStack(alignment: .leading, spacing: 4) {
                Text(isFace ? "The part that gets enrolled" : cropCaption)
                    .font(.footnote)
                    .foregroundStyle(Theme.textPrimary)
                QualityBar(quality: candidate.quality)
                    .frame(width: 120)
            }
            Spacer(minLength: 0)
        }
    }

    private var cropCaption: String {
        candidate.plate.isEmpty ? "No plate text was read" : "Read as \(candidate.plate)"
    }

    // MARK: Facts

    private var facts: some View {
        VStack(alignment: .leading, spacing: 6) {
            fact("Camera", candidate.camera)
            fact("Seen", Self.seenFormatter.string(
                from: Date(timeIntervalSince1970: candidate.createdAt)
            ))
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(RoundedRectangle(cornerRadius: 12).fill(Theme.surface))
    }

    private func fact(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label).font(.footnote).foregroundStyle(Theme.textSecondary)
            Spacer()
            Text(value).font(.footnote).foregroundStyle(Theme.textPrimary)
        }
    }

    private static let seenFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateStyle = .medium
        f.timeStyle = .short
        return f
    }()

    // MARK: Add to

    private var addTo: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(isFace ? "Add to a person" : "Add to a vehicle")
                .font(.headline)
                .foregroundStyle(Theme.textPrimary)

            if profiles.isEmpty {
                Text(isFace
                     ? "No people yet — add one on the previous screen first."
                     : "No vehicles yet — add one on the previous screen first.")
                    .font(.footnote)
                    .foregroundStyle(Theme.textSecondary)
            } else {
                ForEach(profiles) { profile in
                    Button {
                        Task {
                            working = true
                            await onEnroll(profile)
                            working = false
                            dismiss()
                        }
                    } label: {
                        HStack {
                            Text(profile.name).foregroundStyle(Theme.textPrimary)
                            Spacer()
                            if working {
                                ProgressView().tint(Theme.accent)
                            } else {
                                Image(systemName: "plus.circle")
                                    .foregroundStyle(Theme.accent)
                            }
                        }
                        .padding(.vertical, 10)
                        .padding(.horizontal, 14)
                        .background(RoundedRectangle(cornerRadius: 10).fill(Theme.surface))
                    }
                    .disabled(working)
                }
            }

            Text("Adding this shot appends it as a reference. Nothing is trained and nothing is overwritten — remove the sample later and it is as if you never added it.")
                .font(.caption)
                .foregroundStyle(Theme.textSecondary)
        }
    }
}
