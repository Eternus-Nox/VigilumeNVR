import SwiftUI

/// One profile: its enrolled references, and the controls that decide how
/// readily it matches.
///
/// The screen leans on one fact that is easy to get wrong about this feature:
/// there is NO TRAINING STEP. Enrolling copies a reference into the gallery and
/// deleting one removes it; nothing on the server is fitted, so every action
/// here is instant and completely reversible. That is why references can be
/// deleted freely and why the copy says "reference", never "training data" —
/// the mental model of "I might ruin the model" would make people hoard bad
/// shots, which is the one thing that actually hurts accuracy.
struct RecognitionProfileDetailView: View {
    let profileId: Int
    var onChange: () async -> Void

    @EnvironmentObject private var session: SessionModel
    @Environment(\.dismiss) private var dismiss

    @State private var detail: RecognitionProfileDetail?
    @State private var loading = true
    @State private var confirmingDelete = false
    @State private var newPlate = ""

    /// ONE alert modifier, switched by this. Two `.alert` modifiers on one view
    /// is a SwiftUI trap — only one ever presents — so the plate prompt and the
    /// error alert would silently swallow each other. (The confirmation dialog
    /// below is a different modifier and coexists fine.)
    private enum ActiveAlert: Identifiable {
        case plate
        case error(String)

        var id: String {
            switch self {
            case .plate: return "plate"
            case .error(let message): return "error:\(message)"
            }
        }
    }
    @State private var activeAlert: ActiveAlert?
    @State private var enabled = true
    @State private var strictness: Double = 0

    private var isPerson: Bool { detail?.kind == "person" }

    private var activeAlertTitle: String {
        guard let activeAlert else { return "Something went wrong" }
        switch activeAlert {
        case .plate: return "Add a plate"
        case .error: return "Something went wrong"
        }
    }

    private let columns = [GridItem(.adaptive(minimum: 96), spacing: 8)]

    var body: some View {
        List {
            if let detail {
                referencesSection(detail)
                settingsSection(detail)
                dangerSection(detail)
            }
        }
        .listStyle(.insetGrouped)
        .scrollContentBackground(.hidden)
        .background(Theme.bg)
        .navigationTitle(detail?.name ?? "Profile")
        .navigationBarTitleDisplayMode(.inline)
        .overlay {
            if loading && detail == nil { ProgressView().tint(Theme.accent) }
        }
        .alert(
            activeAlertTitle,
            isPresented: Binding(
                get: { activeAlert != nil },
                set: { if !$0 { activeAlert = nil } }
            ),
            presenting: activeAlert
        ) { alert in
            switch alert {
            case .plate:
                TextField("Plate", text: $newPlate)
                    .textInputAutocapitalization(.characters)
                    .autocorrectionDisabled()
                Button("Cancel", role: .cancel) {}
                Button("Add") { Task { await addPlate() } }
            case .error:
                Button("OK", role: .cancel) {}
            }
        } message: { alert in
            switch alert {
            case .plate:
                Text("Spaces and dashes are ignored. A vehicle can carry more than one plate.")
            case .error(let message):
                Text(message)
            }
        }
        .confirmationDialog(
            "Delete \(detail?.name ?? "this profile")?",
            isPresented: $confirmingDelete,
            titleVisibility: .visible
        ) {
            Button("Delete", role: .destructive) { Task { await deleteProfile() } }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text(isPerson
                 ? "Their enrolled face images are deleted from the server too. This can't be undone."
                 : "Its plates and any enrolled images are deleted from the server too. This can't be undone.")
        }
        .task { await reload() }
    }

    // MARK: Sections

    @ViewBuilder
    private func referencesSection(_ detail: RecognitionProfileDetail) -> some View {
        Section {
            if detail.samples.isEmpty {
                Text(isPerson
                     ? "No faces enrolled. Open Unknown Faces and pick the clearest sightings of them."
                     : "No plates yet. Add one below, or enroll a sighting from Unread Plates.")
                    .font(.callout)
                    .foregroundStyle(Theme.textSecondary)
                    .listRowBackground(Theme.surface)
            } else if isPerson {
                LazyVGrid(columns: columns, spacing: 8) {
                    ForEach(detail.samples) { sample in
                        sampleTile(sample)
                    }
                }
                .listRowBackground(Theme.surface)
                .listRowInsets(EdgeInsets(top: 10, leading: 12, bottom: 10, trailing: 12))
            } else {
                ForEach(detail.samples) { sample in
                    HStack {
                        Text(sample.plate.isEmpty ? "(image only)" : sample.plate)
                            .font(.system(.body, design: .monospaced))
                            .foregroundStyle(Theme.textPrimary)
                        Spacer()
                        Button(role: .destructive) {
                            Task { await deleteSample(sample.id) }
                        } label: {
                            Image(systemName: "trash")
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(Theme.dangerSoft)
                    }
                    .listRowBackground(Theme.surface)
                }
            }

            if !isPerson {
                Button {
                    newPlate = ""
                    activeAlert = .plate
                } label: {
                    Label("Add a plate", systemImage: "plus")
                }
                .listRowBackground(Theme.surface)
            }
        } header: {
            Text(isPerson ? "Enrolled faces" : "Plates")
        } footer: {
            if isPerson {
                Text("Accuracy comes from VARIETY, not volume — a few shots across different angles, lighting and seasons beat a dozen of the same pose. Tap any reference to remove it; nothing is retrained, so removing one takes effect immediately.")
            } else {
                Text("Plates are matched allowing for one misread character, and letters that look like digits (O/0, I/1, S/5) are treated as the same.")
            }
        }
    }

    @ViewBuilder
    private func sampleTile(_ sample: RecognitionSample) -> some View {
        ZStack(alignment: .topTrailing) {
            AsyncImage(url: session.api?.recognitionSampleImageURL(id: sample.id)) { phase in
                switch phase {
                case .success(let image):
                    image.resizable().scaledToFill()
                default:
                    Rectangle().fill(Theme.bgDeep)
                        .overlay { Image(systemName: "person.fill").foregroundStyle(Theme.textSecondary) }
                }
            }
            .frame(width: 96, height: 96)
            .clipShape(RoundedRectangle(cornerRadius: 8))

            Button(role: .destructive) {
                Task { await deleteSample(sample.id) }
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .font(.body)
                    .foregroundStyle(.white, Color.black.opacity(0.6))
                    .padding(3)
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Remove this reference")
        }
    }

    @ViewBuilder
    private func settingsSection(_ detail: RecognitionProfileDetail) -> some View {
        Section {
            Toggle("Match this profile", isOn: $enabled)
                .tint(Theme.accent)
                .listRowBackground(Theme.surface)
                .onChange(of: enabled) { _, newValue in
                    Task { await patch(enabled: newValue) }
                }

            if isPerson {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Strictness").foregroundStyle(Theme.textPrimary)
                        Spacer()
                        Text(strictnessLabel)
                            .font(.caption)
                            .foregroundStyle(Theme.textSecondary)
                    }
                    Slider(value: $strictness, in: 0...1, step: 0.05) { editing in
                        // 0 means "inherit the server default", which has to be
                        // sent as an explicit null. Omitting the field would
                        // mean "leave it alone" to the server's partial-update
                        // semantics, so dragging back to Default would appear
                        // to work and change nothing.
                        if !editing {
                            Task {
                                await patch(threshold: strictness == 0 ? .useDefault
                                                                       : .value(strictness))
                            }
                        }
                    }
                    .tint(Theme.accent)
                }
                .listRowBackground(Theme.surface)
            }
        } header: {
            Text("Matching")
        } footer: {
            if isPerson {
                Text("Leave strictness at Default unless this person keeps being confused with someone else — raising it affects only them, not everyone else's recognition.")
            } else {
                Text("Turning a profile off keeps it and its plates, but stops it matching.")
            }
        }
    }

    private var strictnessLabel: String {
        if strictness == 0 { return "Default" }
        if strictness < 0.35 { return "Loose" }
        if strictness < 0.55 { return "Normal" }
        if strictness < 0.75 { return "Strict" }
        return "Very strict"
    }

    @ViewBuilder
    private func dangerSection(_ detail: RecognitionProfileDetail) -> some View {
        Section {
            Button(role: .destructive) {
                confirmingDelete = true
            } label: {
                Label("Delete profile", systemImage: "trash")
            }
            .listRowBackground(Theme.surface)
        }
    }

    // MARK: Actions

    private func reload() async {
        guard let api = session.api else { return }
        loading = true
        defer { loading = false }
        do {
            let d = try await api.recognitionProfile(id: profileId)
            detail = d
            enabled = d.enabled
            strictness = d.threshold ?? 0
        } catch {
            // A cancelled request is not a failure — SwiftUI cancels the
            // .task when the view refreshes. Alerting on it turns pull-to-
            // refresh into a scary "is the NVR reachable?".
            if (error as? ApiError)?.isCancelled == true { return }
            activeAlert = .error((error as? ApiError)?.message ?? error.localizedDescription)
        }
    }

    private func patch(
        enabled: Bool? = nil,
        threshold: APIClient.ThresholdPatch = .unchanged
    ) async {
        guard let api = session.api else { return }
        do {
            detail = try await api.updateRecognitionProfile(
                id: profileId, enabled: enabled, threshold: threshold
            )
            await onChange()
        } catch {
            // A cancelled request is not a failure — SwiftUI cancels the
            // .task when the view refreshes. Alerting on it turns pull-to-
            // refresh into a scary "is the NVR reachable?".
            if (error as? ApiError)?.isCancelled == true { return }
            activeAlert = .error((error as? ApiError)?.message ?? error.localizedDescription)
            await reload()   // put the controls back where the server says they are
        }
    }

    private func addPlate() async {
        let plate = newPlate.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !plate.isEmpty, let api = session.api else { return }
        do {
            _ = try await api.addPlateSample(profileId: profileId, plate: plate)
            await reload()
            await onChange()
        } catch {
            // A cancelled request is not a failure — SwiftUI cancels the
            // .task when the view refreshes. Alerting on it turns pull-to-
            // refresh into a scary "is the NVR reachable?".
            if (error as? ApiError)?.isCancelled == true { return }
            activeAlert = .error((error as? ApiError)?.message ?? error.localizedDescription)
        }
    }

    private func deleteSample(_ id: Int) async {
        guard let api = session.api else { return }
        do {
            try await api.deleteRecognitionSample(id: id)
            await reload()
            await onChange()
        } catch {
            // A cancelled request is not a failure — SwiftUI cancels the
            // .task when the view refreshes. Alerting on it turns pull-to-
            // refresh into a scary "is the NVR reachable?".
            if (error as? ApiError)?.isCancelled == true { return }
            activeAlert = .error((error as? ApiError)?.message ?? error.localizedDescription)
        }
    }

    private func deleteProfile() async {
        guard let api = session.api else { return }
        do {
            try await api.deleteRecognitionProfile(id: profileId)
            await onChange()
            dismiss()
        } catch {
            // A cancelled request is not a failure — SwiftUI cancels the
            // .task when the view refreshes. Alerting on it turns pull-to-
            // refresh into a scary "is the NVR reachable?".
            if (error as? ApiError)?.isCancelled == true { return }
            activeAlert = .error((error as? ApiError)?.message ?? error.localizedDescription)
        }
    }
}
