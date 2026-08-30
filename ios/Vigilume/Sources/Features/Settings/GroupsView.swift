import SwiftUI

/// Settings › Camera groups: the iOS twin of the web's Settings → Groups tab
/// (frontend/src/pages/settings/GroupsTab.tsx). Create, rename, re-member and
/// delete the named camera subsets that drive the dashboard's filter chips (see
/// `CamerasView`) and the web's TV mode.
///
/// READ-ONLY FOR A VIEWER. Reading `/api/groups` is any-auth — a viewer needs
/// the filter chips to navigate the cameras they may watch — but creating,
/// renaming, re-membering and deleting are admin. Groups are SHARED: one
/// account's edit changes what every other account sees, so they are
/// configuration, not a per-user convenience.
///
/// This screen therefore stays outside `SettingsHomeView`'s `session.isAdmin`
/// gate (a viewer still opens it) and hides its own editing controls instead.
/// The backend enforces the rule; hiding the controls just avoids offering a
/// viewer buttons that would 403.
///
/// A group's camera list is an ORDER, not a set: it's the order the dashboard
/// lays the tiles out in, so members are reorderable. The backend tolerates
/// stale names in storage and filters unknown ones out of every response, so a
/// camera deleted and recreated under the same name reappears in its groups.
///
/// Reached from `SettingsHomeView`, so it lives inside that view's existing
/// NavigationStack — no NavigationStack of its own.
struct GroupsView: View {
    @EnvironmentObject private var session: SessionModel

    @State private var groups: [CameraGroup] = []
    @State private var cameras: [Camera] = []
    @State private var loading = true
    @State private var errorMessage: String?
    @State private var showingAdd = false
    /// Set while the delete confirmation for a group is up.
    @State private var pendingDelete: CameraGroup?
    @State private var busy = false

    var body: some View {
        Group {
            if let errorMessage, groups.isEmpty {
                ContentUnavailableView(
                    "Couldn't load groups",
                    systemImage: "rectangle.3.group",
                    description: Text(errorMessage)
                )
            } else if groups.isEmpty && !loading {
                ContentUnavailableView(
                    "No groups",
                    systemImage: "rectangle.3.group",
                    description: Text("Tap + to group cameras into a dashboard filter.")
                )
            } else {
                list
            }
        }
        .background(Theme.bg)
        .navigationTitle("Camera groups")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            if session.isAdmin {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        showingAdd = true
                    } label: {
                        Image(systemName: "plus")
                    }
                    .accessibilityLabel("Add group")
                    .disabled(busy)
                }
            }
        }
        .sheet(isPresented: $showingAdd) {
            AddGroupSheet(cameras: cameras) { created in
                groups.append(created)
            }
        }
        // Naming the group in the prompt mirrors the web's ConfirmDialog rather
        // than relying on a bare swipe-to-delete.
        .alert(
            "Delete group",
            isPresented: Binding(
                get: { pendingDelete != nil },
                set: { if !$0 { pendingDelete = nil } }
            ),
            presenting: pendingDelete
        ) { group in
            Button("Delete", role: .destructive) {
                Task { await delete(group) }
            }
            Button("Cancel", role: .cancel) { pendingDelete = nil }
        } message: { group in
            Text("Delete the group “\(group.name)”? The cameras themselves aren't affected.")
        }
        .task { await load() }
        .refreshable { await load() }
    }

    // MARK: List

    private var list: some View {
        List {
            Section {
                ForEach(groups) { group in
                    NavigationLink {
                        GroupDetailView(group: group, cameras: cameras) { updated in
                            if let index = groups.firstIndex(where: { $0.id == updated.id }) {
                                groups[index] = updated
                            }
                        }
                    } label: {
                        row(group)
                    }
                    .listRowBackground(Theme.surface)
                }
                // No swipe-to-delete for a viewer: the gesture would look
                // available and every commit would 403.
                .onDelete(perform: session.isAdmin ? confirmDelete : nil)
            } footer: {
                Text(
                    session.isAdmin
                    ? "Groups are shared with everyone who signs in to this server — they're a property of the NVR, not of your account. They show up as filter chips on the Cameras tab. Swipe a row to delete a group."
                    : "Groups are shared with everyone who signs in to this server. They show up as filter chips on the Cameras tab. Only an administrator can add, rename or change them."
                )
            }
        }
        .listStyle(.plain)
        .scrollContentBackground(.hidden)
        .overlay {
            if loading && groups.isEmpty {
                ProgressView().tint(Theme.accent)
            }
        }
    }

    private func row(_ group: CameraGroup) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(group.name)
                .font(.subheadline.weight(.medium))
                .foregroundStyle(Theme.textPrimary)
            Text(memberSummary(group))
                .font(.caption)
                .foregroundStyle(Theme.textSecondary)
                .lineLimit(1)
        }
        .padding(.vertical, 4)
    }

    /// "3 cameras · Front Yard, Driveway, Porch" — the count plus as many names
    /// as fit, so a row is identifiable without opening it.
    private func memberSummary(_ group: CameraGroup) -> String {
        let count = group.cameras.count
        guard count > 0 else { return "No cameras" }
        let names = group.cameras.map { displayName(for: $0) }.joined(separator: ", ")
        return "\(count) camera\(count == 1 ? "" : "s") · \(names)"
    }

    /// Friendly name, falling back to the prettified slug (front_yard → Front
    /// Yard) — the same fallback the rest of the app uses.
    private func displayName(for name: String) -> String {
        if let match = cameras.first(where: { $0.name == name }), !match.friendlyName.isEmpty {
            return match.friendlyName
        }
        return name.replacingOccurrences(of: "_", with: " ").capitalized
    }

    // MARK: Loading

    private func load() async {
        guard let api = session.api else { return }
        loading = groups.isEmpty
        // Cameras are only needed to label members; a failure there must not
        // stop the groups themselves from loading (names degrade to the slug).
        if let fetched = try? await api.cameras() {
            cameras = fetched
        }
        do {
            groups = try await api.groups().sorted { $0.position < $1.position }
            errorMessage = nil
        } catch {
            session.handleAPIError(error)
            errorMessage = (error as? ApiError)?.message ?? error.localizedDescription
        }
        loading = false
    }

    // MARK: Delete

    /// Swipe-to-delete only ARMS the confirmation — the row stays until the
    /// alert is confirmed.
    private func confirmDelete(at offsets: IndexSet) {
        guard let index = offsets.first, groups.indices.contains(index) else { return }
        pendingDelete = groups[index]
    }

    private func delete(_ group: CameraGroup) async {
        guard let api = session.api else { return }
        busy = true
        defer { busy = false }
        pendingDelete = nil
        do {
            try await api.deleteGroup(id: group.id)
            groups.removeAll { $0.id == group.id }
            errorMessage = nil
        } catch {
            session.handleAPIError(error)
            errorMessage = "Delete failed: "
                + ((error as? ApiError)?.message ?? error.localizedDescription)
        }
    }
}

// MARK: - Group detail (rename + members)

/// One group's editor: rename it, reorder or remove its cameras, and add the
/// ones it doesn't have yet. Every change commits immediately (PUT
/// /api/groups/{id} with just the edited key) and reverts on failure — there's
/// no Save button to strand a half-applied edit.
private struct GroupDetailView: View {
    let cameras: [Camera]
    /// Called with the server's version after each successful commit so the list
    /// behind us stays truthful.
    let onChanged: (CameraGroup) -> Void

    @EnvironmentObject private var session: SessionModel

    @State private var group: CameraGroup
    @State private var draftName: String
    @State private var busy = false
    @State private var errorMessage: String?

    init(group: CameraGroup, cameras: [Camera], onChanged: @escaping (CameraGroup) -> Void) {
        self.cameras = cameras
        self.onChanged = onChanged
        _group = State(initialValue: group)
        _draftName = State(initialValue: group.name)
    }

    /// Cameras not yet in the group, in dashboard order.
    private var available: [Camera] {
        cameras.filter { !group.cameras.contains($0.name) }
    }

    /// The group's members as cameras, in the group's stored order. Names with
    /// no camera are dropped — the backend already filters them out of
    /// responses, so this only matters if the two lists raced.
    private var members: [Camera] {
        group.cameras.compactMap { name in cameras.first { $0.name == name } }
    }

    var body: some View {
        Form {
            Section {
                TextField("Front yard", text: $draftName)
                    .foregroundStyle(Theme.textPrimary)
                    // Commit on blur/return rather than per keystroke.
                    .onSubmit { Task { await rename() } }
                    .submitLabel(.done)
                    // A viewer reads the name; renaming is admin (shared config).
                    .disabled(busy || !session.isAdmin)
            } header: {
                Text("Name")
            } footer: {
                Text("Up to 64 characters, and unique across the server.")
            }
            .listRowBackground(Theme.surface)

            Section {
                if members.isEmpty {
                    Text("No cameras in this group yet.")
                        .font(.footnote)
                        .foregroundStyle(Theme.textSecondary)
                } else {
                    ForEach(members) { camera in
                        HStack(spacing: 12) {
                            Text(displayName(camera))
                                .foregroundStyle(Theme.textPrimary)
                            Spacer(minLength: 0)
                        }
                    }
                    .onMove(perform: session.isAdmin ? move : nil)
                    .onDelete(perform: session.isAdmin ? removeMembers : nil)
                }
            } header: {
                Text("Cameras")
            } footer: {
                if !session.isAdmin {
                    Text("This is the order the dashboard lays the tiles out in. Only an administrator can change a group.")
                } else if members.count > 1 {
                    Text("This is the order the dashboard lays the tiles out in — tap Edit to drag, or swipe a row to remove it from the group.")
                } else if !members.isEmpty {
                    Text("Swipe a row to remove it from the group. The camera itself isn't deleted.")
                }
            }
            .listRowBackground(Theme.surface)

            if session.isAdmin, !available.isEmpty {
                Section {
                    ForEach(available) { camera in
                        Button {
                            Task { await addMember(camera) }
                        } label: {
                            Label(displayName(camera), systemImage: "plus.circle.fill")
                                .foregroundStyle(Theme.accent)
                        }
                        .disabled(busy)
                    }
                } header: {
                    Text("Add a camera")
                } footer: {
                    Text("Added cameras go to the end of the order.")
                }
                .listRowBackground(Theme.surface)
            }

            if let errorMessage {
                Section {
                    Text(errorMessage)
                        .font(.footnote)
                        .foregroundStyle(Theme.danger)
                }
                .listRowBackground(Theme.surface)
            }
        }
        .scrollContentBackground(.hidden)
        .background(Theme.bg)
        .navigationTitle(group.name)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                if busy {
                    ProgressView().tint(Theme.accent)
                } else if session.isAdmin, !members.isEmpty {
                    EditButton()
                }
            }
        }
        // A rename typed but never submitted still commits when the screen goes
        // — but only for an admin, who is the only one whose field was editable.
        .onDisappear { if session.isAdmin { Task { await rename() } } }
    }

    private func displayName(_ camera: Camera) -> String {
        camera.friendlyName.isEmpty
            ? camera.name.replacingOccurrences(of: "_", with: " ").capitalized
            : camera.friendlyName
    }

    // MARK: Commits

    private func rename() async {
        let trimmed = draftName.trimmingCharacters(in: .whitespacesAndNewlines)
        // A blank name is a 422; treat it as "no edit" and put the old one back
        // rather than bouncing the user off the validator.
        guard !trimmed.isEmpty else {
            draftName = group.name
            return
        }
        guard trimmed != group.name else { return }
        await commit(name: trimmed) { draftName = group.name }
    }

    private func move(from source: IndexSet, to destination: Int) {
        var names = group.cameras
        names.move(fromOffsets: source, toOffset: destination)
        Task { await commit(cameras: names) }
    }

    private func removeMembers(at offsets: IndexSet) {
        let doomed = Set(offsets.map { members[$0].name })
        Task { await commit(cameras: group.cameras.filter { !doomed.contains($0) }) }
    }

    private func addMember(_ camera: Camera) async {
        await commit(cameras: group.cameras + [camera.name])
    }

    /// Apply optimistically, PUT only the edited key, then take the server's
    /// version as truth. On failure the previous group is restored so the screen
    /// never claims an edit that didn't land.
    private func commit(
        name: String? = nil,
        cameras names: [String]? = nil,
        onRevert: (() -> Void)? = nil
    ) async {
        guard let api = session.api else { return }
        let previous = group
        if let name { group.name = name }
        if let names { group.cameras = names }
        busy = true
        defer { busy = false }
        do {
            let saved = try await api.updateGroup(id: group.id, name: name, cameras: names)
            group = saved
            draftName = saved.name
            onChanged(saved)
            errorMessage = nil
        } catch {
            group = previous
            onRevert?()
            session.handleAPIError(error)
            // 409 duplicate name, 404 deleted elsewhere, 422 validation — the
            // backend's detail is the readable one.
            errorMessage = (error as? ApiError)?.message ?? error.localizedDescription
        }
    }
}

// MARK: - Add sheet

/// POST /api/groups — name plus an initial, ordered member list. Selection order
/// IS the display order, exactly like the web's create modal.
private struct AddGroupSheet: View {
    let cameras: [Camera]
    /// Called with the created group so the list behind us updates without a refetch.
    let onCreated: (CameraGroup) -> Void

    @EnvironmentObject private var session: SessionModel
    @Environment(\.dismiss) private var dismiss

    @State private var name = ""
    @State private var members: [String] = []
    @State private var saving = false
    @State private var errorMessage: String?

    private var canSave: Bool {
        !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("Front yard", text: $name)
                } header: {
                    Text("Name")
                } footer: {
                    Text("Up to 64 characters, and unique across the server.")
                }
                .listRowBackground(Theme.surface)

                Section {
                    if cameras.isEmpty {
                        Text("No cameras available yet.")
                            .font(.footnote)
                            .foregroundStyle(Theme.textSecondary)
                    } else {
                        ForEach(cameras) { camera in
                            Button {
                                toggle(camera)
                            } label: {
                                HStack {
                                    Text(displayName(camera))
                                        .foregroundStyle(Theme.textPrimary)
                                    Spacer()
                                    // The tap index doubles as the display
                                    // order, so show it rather than a checkmark.
                                    if let index = members.firstIndex(of: camera.name) {
                                        Text("\(index + 1)")
                                            .font(.caption.weight(.semibold))
                                            .foregroundStyle(Theme.accent)
                                        Image(systemName: "checkmark")
                                            .foregroundStyle(Theme.accent)
                                    }
                                }
                            }
                        }
                    }
                } header: {
                    Text("Cameras")
                } footer: {
                    Text("Tap to add or remove. The numbers are the order the dashboard lays the tiles out in — you can reorder them after creating the group.")
                }
                .listRowBackground(Theme.surface)

                if let errorMessage {
                    Section {
                        Text(errorMessage)
                            .font(.footnote)
                            .foregroundStyle(Theme.danger)
                    }
                    .listRowBackground(Theme.surface)
                }
            }
            .scrollContentBackground(.hidden)
            .background(Theme.bg)
            .navigationTitle("New Group")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                        .disabled(saving)
                }
                ToolbarItem(placement: .confirmationAction) {
                    if saving {
                        ProgressView().tint(Theme.accent)
                    } else {
                        Button("Create") { Task { await save() } }
                            .disabled(!canSave)
                    }
                }
            }
        }
    }

    private func displayName(_ camera: Camera) -> String {
        camera.friendlyName.isEmpty
            ? camera.name.replacingOccurrences(of: "_", with: " ").capitalized
            : camera.friendlyName
    }

    private func toggle(_ camera: Camera) {
        if let index = members.firstIndex(of: camera.name) {
            members.remove(at: index)
        } else {
            members.append(camera.name)
        }
    }

    private func save() async {
        guard let api = session.api, canSave, !saving else { return }
        saving = true
        defer { saving = false }
        do {
            let created = try await api.createGroup(
                name: name.trimmingCharacters(in: .whitespacesAndNewlines),
                cameras: members
            )
            onCreated(created)
            dismiss()
        } catch {
            session.handleAPIError(error)
            // 409 "Group 'x' already exists" is the common one here.
            errorMessage = (error as? ApiError)?.message ?? error.localizedDescription
        }
    }
}
