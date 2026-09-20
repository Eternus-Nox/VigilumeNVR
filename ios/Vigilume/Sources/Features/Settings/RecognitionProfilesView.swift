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
    @State private var settings: SettingsDocument.Recognition?
    @State private var savingSettings = false
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

    /// Footer copy for the Recognition section.
    ///
    /// A PLAIN String-returning function, deliberately, rather than the obvious
    /// inline version. A ternary whose two branches are both interpolated
    /// strings, inside a `Text(...)`, inside a ViewBuilder closure, is the exact
    /// shape that makes Swift's type checker give up:
    ///
    ///     error: The compiler is unable to type-check this expression in
    ///            reasonable time; try breaking up the expression
    ///
    /// The failure is nastier than it sounds. The type never gets built, so
    /// every USE SITE reports "Cannot find 'RecognitionProfilesView' in scope"
    /// — which points at SettingsHomeView, a file with nothing wrong in it.
    ///
    /// Returning `String` rather than `Text` keeps this out of the ViewBuilder
    /// entirely: ordinary control flow the compiler checks in one pass, and the
    /// interpolation happens once, into a local, instead of inside an
    /// expression it has to solve overloads across.
    private func recognitionFooter(_ settings: SettingsDocument.Recognition) -> String {
        guard settings.enabled else {
            return """
                Off. Turning this on downloads two extra models (~41 MB) and keeps face \
                images on the server for \(settings.candidateRetentionDays) days so you \
                can enroll people afterwards. Profiles can be set up either way — they \
                start matching once this is on.
                """
        }
        let hold = Int(settings.notifyGraceSeconds)
        if settings.notifyMode == "unknown_only" {
            return """
                Enrolled people and vehicles arrive silently; anyone else still alerts — \
                including someone nobody could identify. Alerts are held about \(hold)s \
                while recognition decides.
                """
        }
        return """
            Alerts name a recognized person or vehicle. Held about \(hold)s while \
            recognition decides, then sent either way.
            """
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

            if let settings {
                Section {
                    // `newValue in` is NOT optional here. `$0` inside `Task { }`
                    // binds to the TASK's closure, which takes no arguments —
                    // so the setter appears to ignore its own parameter, Swift
                    // falls back to the (Value, Transaction) overload of
                    // Binding.init(get:set:), and the error talks about 2
                    // arguments and an `@isolated(any) () async -> ()` that
                    // nothing in this code mentions. Name the parameter.
                    Toggle("Recognize faces & plates", isOn: Binding(
                        get: { settings.enabled },
                        set: { newValue in Task { await setEnabled(newValue) } }
                    ))
                    .tint(Theme.accent)
                    .disabled(savingSettings)
                    .listRowBackground(Theme.surface)

                    if settings.enabled {
                        Picker("Alert me about", selection: Binding(
                            get: { settings.notifyMode },
                            set: { newValue in Task { await setNotifyMode(newValue) } }
                        )) {
                            Text("Everyone").tag("all")
                            Text("Only unrecognized").tag("unknown_only")
                        }
                        .disabled(savingSettings)
                        .listRowBackground(Theme.surface)
                    }
                } header: {
                    Text("Recognition")
                } footer: {
                    Text(recognitionFooter(settings))
                }
            }

            if let status, let settings, settings.enabled, !status.ready {
                Section {
                    Label {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Models still loading")
                                .foregroundStyle(Theme.textPrimary)
                            Text("The server is fetching the recognition models. This takes a moment on first use, and recognition starts by itself once they are ready.")
                                .font(.caption)
                                .foregroundStyle(Theme.textSecondary)
                        }
                    } icon: {
                        Image(systemName: "arrow.down.circle")
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
            // Separate do/catch: an older backend without a settings document
            // must not blank the profile list it just returned successfully.
            settings = try? await api.settingsDocument().recognition
        } catch {
            // A cancelled request is not a failure — SwiftUI cancels the
            // .task when the view refreshes. Alerting on it turns pull-to-
            // refresh into a scary "is the NVR reachable?".
            if (error as? ApiError)?.isCancelled == true { return }
            activeAlert = .error((error as? ApiError)?.message ?? error.localizedDescription)
        }
    }

    private func setEnabled(_ on: Bool) async {
        await patchRecognition(.init(enabled: on))
    }

    private func setNotifyMode(_ mode: String) async {
        await patchRecognition(.init(notifyMode: mode))
    }

    /// Send ONLY the field that changed. Every field on the patch is optional
    /// and nil is omitted, so flipping the toggle cannot reset the retention
    /// window or the alert mode.
    private func patchRecognition(_ patch: SettingsPatch.Recognition) async {
        guard let api = session.api else { return }
        savingSettings = true
        defer { savingSettings = false }
        do {
            var body = SettingsPatch()
            body.recognition = patch
            settings = try await api.patchSettings(body).recognition
            // The server loads or releases the models on its own schedule, so
            // re-read status rather than assuming the toggle took effect now.
            status = try? await api.recognitionStatus()
        } catch {
            // A cancelled request is not a failure — SwiftUI cancels the
            // .task when the view refreshes. Alerting on it turns pull-to-
            // refresh into a scary "is the NVR reachable?".
            if (error as? ApiError)?.isCancelled == true { return }
            activeAlert = .error((error as? ApiError)?.message ?? error.localizedDescription)
            settings = try? await api.settingsDocument().recognition
        }
    }

    private func create() async {
        let name = newName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty, let api = session.api else { return }
        do {
            _ = try await api.createRecognitionProfile(kind: kind, name: name)
            await reload()
        } catch {
            // A cancelled request is not a failure — SwiftUI cancels the
            // .task when the view refreshes. Alerting on it turns pull-to-
            // refresh into a scary "is the NVR reachable?".
            if (error as? ApiError)?.isCancelled == true { return }
            activeAlert = .error((error as? ApiError)?.message ?? error.localizedDescription)
        }
    }
}
