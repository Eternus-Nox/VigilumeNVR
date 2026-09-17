import SwiftUI

/// Unknown Faces / Unread Plates — the picker you enroll FROM.
///
/// This is the screen that answers "which image should we use". The server has
/// already done the part a person cannot do by eye at thumbnail size: for every
/// tracked face or plate it kept several shots spread across the sighting and
/// scored each for LEGIBILITY — sharpness, size, exposure, and pose — which is
/// a different question from the detector's "is something there". A big,
/// centred, motion-blurred face scores very well on the second and is useless
/// for the first.
///
/// So the list arrives best-first and every tile states its own quality. The
/// operator is choosing between genuinely good candidates rather than hunting
/// through frames, and the one thing they are asked to judge is the thing only
/// a person can: whether this is actually who they think it is.
struct RecognitionCandidatesView: View {
    /// "face" | "plate".
    let kind: String
    let profiles: [RecognitionProfile]
    var onChange: () async -> Void

    @EnvironmentObject private var session: SessionModel

    @State private var candidates: [RecognitionCandidate] = []
    @State private var selected: Set<Int> = []
    @State private var loading = true
    @State private var errorMessage: String?
    @State private var enrolling = false
    @State private var showingPicker = false
    @State private var confirmingClear = false
    @State private var toast: String?

    private var isFace: Bool { kind == "face" }
    private let columns = [GridItem(.adaptive(minimum: 104), spacing: 10)]

    var body: some View {
        ScrollView {
            if candidates.isEmpty && !loading {
                ContentUnavailableView(
                    isFace ? "No unknown faces" : "No unread plates",
                    systemImage: "person.crop.square.badge.camera",
                    description: Text(isFace
                        ? "Faces that don't match anyone enrolled show up here so you can add them."
                        : "Plates that don't match a vehicle show up here.")
                )
                .padding(.top, 60)
            } else {
                LazyVGrid(columns: columns, spacing: 10) {
                    ForEach(candidates) { candidate in
                        tile(candidate)
                    }
                }
                .padding(12)
            }
        }
        .background(Theme.bg)
        .navigationTitle(isFace ? "Unknown Faces" : "Unread Plates")
        .navigationBarTitleDisplayMode(.inline)
        .overlay {
            if loading && candidates.isEmpty { ProgressView().tint(Theme.accent) }
        }
        .safeAreaInset(edge: .bottom) {
            if !selected.isEmpty {
                enrollBar
            }
        }
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Menu {
                    if !selected.isEmpty {
                        Button("Deselect all") { selected.removeAll() }
                    }
                    Button(role: .destructive) {
                        confirmingClear = true
                    } label: {
                        Label(isFace ? "Clear all unknown faces" : "Clear all unread plates",
                              systemImage: "trash")
                    }
                } label: {
                    Image(systemName: "ellipsis.circle")
                }
            }
        }
        .confirmationDialog(
            isFace ? "Clear every unknown face?" : "Clear every unread plate?",
            isPresented: $confirmingClear,
            titleVisibility: .visible
        ) {
            Button("Clear", role: .destructive) { Task { await clearAll() } }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("The images are deleted from the server. Anything already enrolled into a profile is kept.")
        }
        .sheet(isPresented: $showingPicker) {
            profilePicker
        }
        .alert(
            "Something went wrong",
            isPresented: Binding(
                get: { errorMessage != nil },
                set: { if !$0 { errorMessage = nil } }
            )
        ) {
            Button("OK", role: .cancel) { errorMessage = nil }
        } message: {
            Text(errorMessage ?? "")
        }
        .overlay(alignment: .top) {
            if let toast {
                Text(toast)
                    .font(.footnote.weight(.medium))
                    .foregroundStyle(Theme.textPrimary)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
                    .background(Capsule().fill(Theme.surface))
                    .padding(.top, 8)
                    .transition(.move(edge: .top).combined(with: .opacity))
            }
        }
        .task { await reload() }
        .refreshable { await reload() }
    }

    // MARK: Tile

    @ViewBuilder
    private func tile(_ candidate: RecognitionCandidate) -> some View {
        let isSelected = selected.contains(candidate.id)
        VStack(spacing: 4) {
            ZStack(alignment: .topTrailing) {
                AsyncImage(url: session.api?.recognitionCandidateImageURL(id: candidate.id)) { phase in
                    switch phase {
                    case .success(let image):
                        image.resizable().scaledToFill()
                    case .failure:
                        Rectangle().fill(Theme.bgDeep)
                            .overlay { Image(systemName: "photo").foregroundStyle(Theme.textSecondary) }
                    default:
                        Rectangle().fill(Theme.bgDeep)
                    }
                }
                .frame(width: 104, height: 104)
                .clipShape(RoundedRectangle(cornerRadius: 10))
                .overlay {
                    RoundedRectangle(cornerRadius: 10)
                        .strokeBorder(isSelected ? Theme.accent : Color.clear, lineWidth: 3)
                }

                Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                    .font(.title3)
                    .foregroundStyle(isSelected ? Theme.accent : Color.white.opacity(0.8),
                                     Color.black.opacity(0.45))
                    .padding(4)
            }

            // The quality bar is the whole point of this screen: it is the
            // server's legibility score, so a person can pick the readable shot
            // without squinting at a 104 pt thumbnail.
            QualityBar(quality: candidate.quality)
                .frame(width: 104)

            if !candidate.plate.isEmpty {
                Text(candidate.plate)
                    .font(.caption2.monospaced())
                    .foregroundStyle(Theme.textPrimary)
                    .lineLimit(1)
            }
            Text(candidate.camera)
                .font(.caption2)
                .foregroundStyle(Theme.textSecondary)
                .lineLimit(1)
        }
        .contentShape(Rectangle())
        .onTapGesture {
            if isSelected { selected.remove(candidate.id) } else { selected.insert(candidate.id) }
        }
        .contextMenu {
            Button(role: .destructive) {
                Task { await deleteOne(candidate.id) }
            } label: {
                Label("Delete this sighting", systemImage: "trash")
            }
        }
    }

    // MARK: Enroll bar + picker

    private var enrollBar: some View {
        HStack {
            Text("\(selected.count) selected")
                .font(.subheadline)
                .foregroundStyle(Theme.textSecondary)
            Spacer()
            Button {
                showingPicker = true
            } label: {
                if enrolling {
                    ProgressView().tint(.white)
                } else {
                    Text("Enroll…").fontWeight(.semibold)
                }
            }
            .buttonStyle(.borderedProminent)
            .tint(Theme.accent)
            .disabled(enrolling)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(.ultraThinMaterial)
    }

    private var profilePicker: some View {
        NavigationStack {
            List {
                if profiles.isEmpty {
                    Text(isFace
                         ? "No people yet — add one on the previous screen first."
                         : "No vehicles yet — add one on the previous screen first.")
                        .foregroundStyle(Theme.textSecondary)
                        .listRowBackground(Theme.surface)
                }
                ForEach(profiles) { profile in
                    Button {
                        showingPicker = false
                        Task { await enroll(into: profile) }
                    } label: {
                        HStack {
                            Text(profile.name).foregroundStyle(Theme.textPrimary)
                            Spacer()
                            Text("\(profile.sampleCount)")
                                .font(.caption)
                                .foregroundStyle(Theme.textSecondary)
                        }
                    }
                    .listRowBackground(Theme.surface)
                }
            }
            .listStyle(.insetGrouped)
            .scrollContentBackground(.hidden)
            .background(Theme.bg)
            .navigationTitle("Enroll into…")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Cancel") { showingPicker = false }
                }
            }
        }
        .presentationDetents([.medium, .large])
    }

    // MARK: Actions

    private func reload() async {
        guard let api = session.api else { return }
        loading = true
        defer { loading = false }
        do {
            candidates = try await api.recognitionCandidates(kind: kind)
            // Drop selections whose candidate is gone, so the count in the
            // enroll bar can never name rows that no longer exist.
            selected = selected.intersection(Set(candidates.map(\.id)))
        } catch {
            errorMessage = (error as? ApiError)?.message ?? error.localizedDescription
        }
    }

    private func enroll(into profile: RecognitionProfile) async {
        guard let api = session.api, !selected.isEmpty else { return }
        enrolling = true
        defer { enrolling = false }
        do {
            let n = try await api.enrollCandidates(
                profileId: profile.id, candidateIds: Array(selected)
            )
            selected.removeAll()
            await reload()
            await onChange()
            await flash("Enrolled \(n) into \(profile.name)")
        } catch {
            errorMessage = (error as? ApiError)?.message ?? error.localizedDescription
        }
    }

    private func deleteOne(_ id: Int) async {
        guard let api = session.api else { return }
        do {
            try await api.deleteRecognitionCandidate(id: id)
            selected.remove(id)
            await reload()
            await onChange()
        } catch {
            errorMessage = (error as? ApiError)?.message ?? error.localizedDescription
        }
    }

    private func clearAll() async {
        guard let api = session.api else { return }
        do {
            try await api.clearRecognitionCandidates(kind: kind)
            selected.removeAll()
            await reload()
            await onChange()
        } catch {
            errorMessage = (error as? ApiError)?.message ?? error.localizedDescription
        }
    }

    private func flash(_ message: String) async {
        withAnimation { toast = message }
        try? await Task.sleep(nanoseconds: 2_000_000_000)
        withAnimation { toast = nil }
    }
}

/// A small legibility meter. Colour carries the same information as the width,
/// never colour alone — a red/green-only bar is unreadable for a large minority
/// of people, and this bar is the single thing the screen asks you to compare.
struct QualityBar: View {
    let quality: Double

    private var clamped: Double { min(max(quality, 0), 1) }

    private var tint: Color {
        if clamped >= 0.65 { return Theme.success }
        if clamped >= 0.45 { return Theme.warning }
        return Theme.dangerSoft
    }

    private var label: String {
        if clamped >= 0.65 { return "Clear" }
        if clamped >= 0.45 { return "Usable" }
        return "Poor"
    }

    var body: some View {
        VStack(spacing: 2) {
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule().fill(Theme.bgDeep)
                    Capsule().fill(tint).frame(width: geo.size.width * clamped)
                }
            }
            .frame(height: 3)
            Text(label)
                .font(.caption2)
                .foregroundStyle(tint)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Image quality: \(label)")
    }
}
