import SwiftUI

/// Settings › Faces & Plates: the enrolled people and vehicles.
///
/// ADMIN-ONLY, including just LOOKING at it. Every other settings list in this
/// app is gated because it configures the system; this one is gated because of
/// what it contains — a named register of who comes to this address and which
/// cars they drive. The server enforces it (`require_admin` on the whole
/// router, reads included); this view is never reachable for a viewer, so the
/// 403 never has to be explained.
///
/// Reachable from `SettingsHomeView`, so it lives inside that view's existing
/// NavigationStack — no NavigationStack of its own.
struct RecognitionProfilesView: View {
    @EnvironmentObject private var session: SessionModel

    @State private var profiles: [RecognitionProfile] = []
    @State private var status: RecognitionStatus?
    @State private var kind = "person"
    @State private var loading = true
    @State private var newName = ""

    /// ONE alert modifier, switched by this. Two `.alert` modifiers on the same
    /// view is a SwiftUI trap: only one of them ever presents, and which one is
    /// not something you can rely on — so the "add" prompt and the error would
    /// silently swallow each other.
    private enum ActiveAlert: Identifiable {
        case add
        case error(String)

        var id: String {
            switch self {
            case .add: return "add"
            case .error(let message): return "error:\(message)"
            }
        }
    }
    @State private var activeAlert: ActiveAlert?

    private var alertTitle: String {
        guard let activeAlert else { return "Something went wrong" }
        switch activeAlert {
        case .add: return kind == "person" ? "Add a person" : "Add a vehicle"
        case .error: return "Something went wrong"
        }
    }

    private var shown: [RecognitionProfile] {
        profiles.filter { $0.kind == kind }
    }

    var body: some View {
        List {
            Section {
                Picker("Kind", selection: $kind) {
                    Text("People").tag("person")
                    Text("Vehicles").tag("vehicle")
                }
                .pickerStyle(.segmented)
                .listRowBackground(Color.clear)
                .listRowInsets(EdgeInsets(top: 4, leading: 0, bottom: 4, trailing: 0))
            }

            if let status, !status.ready {
                Section {
                    Label {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Recognition is not running")
                                .foregroundStyle(Theme.textPrimary)
                            Text("Profiles can still be set up — they start matching once the recognition model is loaded on the server.")
                                .font(.caption)
                                .foregroundStyle(Theme.textSecondary)
                        }
                    } icon: {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(Theme.warning)
                    }
                    .listRowBackground(Theme.surface)
                }
            }

            if let status, status.staleSamples > 0 {
                Section {
                    Label {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("\(status.staleSamples) reference\(status.staleSamples == 1 ? "" : "s") need re-enrolling")
                                .foregroundStyle(Theme.textPrimary)
                            Text("They were captured with a different recognition model and can no longer be compared.")
                                .font(.caption)
                                .foregroundStyle(Theme.textSecondary)
                        }
                    } icon: {
                        Image(systemName: "arrow.triangle.2.circlepath")
                            .foregroundStyle(Theme.warning)
                    }
                    .listRowBackground(Theme.surface)
                }
            }

            Section {
                if shown.isEmpty && !loading {
                    Text(kind == "person"
                         ? "No people enrolled yet. Add someone, then pick their best shots from Unknown Faces."
                         : "No vehicles enrolled yet. Add one, then type its plate or enroll a sighting.")
                        .font(.callout)
                        .foregroundStyle(Theme.textSecondary)
                        .listRowBackground(Theme.surface)
                }
                ForEach(shown) { profile in
                    NavigationLink {
                        RecognitionProfileDetailView(profileId: profile.id, onChange: reload)
                    } label: {
                        row(profile)
                    }
                    .listRowBackground(Theme.surface)
                }
            } header: {
                Text(kind == "person" ? "People" : "Vehicles")
            }

            Section {
                NavigationLink {
                    RecognitionCandidatesView(
                        kind: kind == "person" ? "face" : "plate",
                        profiles: shown,
                        onChange: reload
                    )
                } label: {
                    Label {
                        HStack {
                            Text(kind == "person" ? "Unknown Faces" : "Unread Plates")
                                .foregroundStyle(Theme.textPrimary)
                            Spacer()
                            let n = status?.candidates[kind == "person" ? "face" : "plate"] ?? 0
                            if n > 0 {
                                Text("\(n)")
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(Theme.textSecondary)
                            }
                        }
                    } icon: {
                        Image(systemName: "person.crop.square.badge.camera")
                            .foregroundStyle(Theme.accent)
                    }
                }
                .listRowBackground(Theme.surface)
            } footer: {
                Text("Sightings that didn't match anyone. Pick the clearest shots to enroll — the server ranks them by how legible they are, not by how sure it was that something was there.")
            }
        }
        .listStyle(.insetGrouped)
        .scrollContentBackground(.hidden)
        .background(Theme.bg)
        .navigationTitle("Faces & Plates")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button { newName = ""; activeAlert = .add } label: {
                    Image(systemName: "plus")
                }
                .accessibilityLabel(kind == "person" ? "Add a person" : "Add a vehicle")
            }
        }
        .alert(
            alertTitle,
            isPresented: Binding(
                get: { activeAlert != nil },
                set: { if !$0 { activeAlert = nil } }
            ),
            presenting: activeAlert
        ) { alert in
            switch alert {
            case .add:
                TextField(kind == "person" ? "Name" : "Vehicle name", text: $newName)
                    .textInputAutocapitalization(.words)
                Button("Cancel", role: .cancel) {}
                Button("Add") { Task { await create() } }
            case .error:
                Button("OK", role: .cancel) {}
            }
        } message: { alert in
            switch alert {
            case .add:
                Text(kind == "person"
                     ? "You'll enroll their face from real sightings afterwards."
                     : "You can type its plate directly, or enroll a sighting.")
            case .error(let message):
                Text(message)
            }
        }
        .overlay {
            if loading && profiles.isEmpty {
                ProgressView().tint(Theme.accent)
            }
        }
        .task { await reload() }
        .refreshable { await reload() }
    }

    @ViewBuilder
    private func row(_ profile: RecognitionProfile) -> some View {
        HStack(spacing: 10) {
            VStack(alignment: .leading, spacing: 2) {
                Text(profile.name)
                    .foregroundStyle(Theme.textPrimary)
                Text(subtitle(profile))
                    .font(.caption)
                    .foregroundStyle(profile.needsReenroll ? Theme.warning : Theme.textSecondary)
            }
            Spacer()
            if !profile.enabled {
                Text("OFF")
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(Theme.textSecondary)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Capsule().fill(Theme.bgDeep))
            }
        }
    }

    private func subtitle(_ profile: RecognitionProfile) -> String {
        if profile.needsReenroll {
            return "Needs re-enrolling"
        }
        if profile.sampleCount == 0 {
            return profile.isPerson ? "No faces enrolled" : "No plate set"
        }
        let unit = profile.isPerson ? "face" : "plate"
        return "\(profile.sampleCount) \(unit)\(profile.sampleCount == 1 ? "" : "s")"
    }

    private func reload() async {
        guard let api = session.api else { return }
        loading = true
        defer { loading = false }
        do {
            async let list = api.recognitionProfiles()
            async let stat = api.recognitionStatus()
            profiles = try await list
            status = try await stat
        } catch {
            activeAlert = .error((error as? ApiError)?.message ?? error.localizedDescription)
        }
    }

    private func create() async {
        let name = newName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty, let api = session.api else { return }
        do {
            _ = try await api.createRecognitionProfile(kind: kind, name: name)
            await reload()
        } catch {
            activeAlert = .error((error as? ApiError)?.message ?? error.localizedDescription)
        }
    }
}
