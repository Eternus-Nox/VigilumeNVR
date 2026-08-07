import SwiftUI

/// Settings › Users (ADMIN ONLY): the iOS twin of the web's Settings → Users tab
/// (frontend/src/pages/settings/UsersTab.tsx). List the managed accounts, create
/// one, reset a password, change a role, and delete.
///
/// The backend (backend/app/routers/users.py) owns every rule; this screen only
/// surfaces them:
///
/// - The whole router is `require_admin`, so we gate on `session.isAdmin` — a
///   viewer that somehow lands here gets a message, not a 403 spinner.
/// - The built-in admin ("admin") is env-controlled (ADMIN_PASSWORD) and has NO
///   DB row: it's never listed, "admin" is a reserved username (400), and it
///   can't be demoted or deleted. It is ALWAYS present, which is why deleting
///   the last DB admin isn't a lockout and isn't blocked.
/// - Usernames must match `^[a-z0-9][a-z0-9_.-]{2,31}$` (3–32 chars) and
///   passwords are 8–256. We hint at both but let the server be the authority.
/// - Demoting the last DB admin is a 400 ("Cannot demote the last admin"). We do
///   NOT pre-empt it client-side — the role change applies optimistically and
///   reverts with the backend's message, exactly like the web.
///
/// Reached from `SettingsHomeView`'s `session.isAdmin` section, so it lives
/// inside that view's existing NavigationStack — no NavigationStack of its own.
struct UsersView: View {
    @EnvironmentObject private var session: SessionModel

    @State private var users: [ManagedUser] = []
    @State private var loading = true
    @State private var errorMessage: String?
    @State private var showingAdd = false
    /// Set while the delete confirmation for a user is up.
    @State private var pendingDelete: ManagedUser?
    @State private var busy = false

    /// The backend's username policy, mirrored for a live hint in the add form.
    /// Advisory only — `_validate_username` is the real gate.
    static let usernamePattern = "^[a-z0-9][a-z0-9_.-]{2,31}$"
    static let minPasswordLength = 8

    var body: some View {
        Group {
            if !session.isAdmin {
                ContentUnavailableView(
                    "Admins only",
                    systemImage: "lock.fill",
                    description: Text("User management is restricted to admin accounts.")
                )
            } else if let errorMessage, users.isEmpty {
                ContentUnavailableView(
                    "Couldn't load users",
                    systemImage: "person.2.slash",
                    description: Text(errorMessage)
                )
            } else if users.isEmpty && !loading {
                ContentUnavailableView(
                    "No additional users",
                    systemImage: "person.2",
                    description: Text("Tap + to create an account. The built-in admin isn't listed here.")
                )
            } else {
                list
            }
        }
        .background(Theme.bg)
        .navigationTitle("Users")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            if session.isAdmin {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        showingAdd = true
                    } label: {
                        Image(systemName: "plus")
                    }
                    .accessibilityLabel("Add user")
                    .disabled(busy)
                }
            }
        }
        .sheet(isPresented: $showingAdd) {
            AddUserSheet { created in
                users.append(created)
                users.sort { $0.username < $1.username }
            }
        }
        // Naming the account in the prompt mirrors the web's ConfirmDialog.
        .alert(
            "Delete user",
            isPresented: Binding(
                get: { pendingDelete != nil },
                set: { if !$0 { pendingDelete = nil } }
            ),
            presenting: pendingDelete
        ) { user in
            Button("Delete", role: .destructive) {
                Task { await delete(user) }
            }
            Button("Cancel", role: .cancel) { pendingDelete = nil }
        } message: { user in
            Text(deleteMessage(for: user))
        }
        .task { await load() }
        .refreshable { await load() }
    }

    /// Deleting yourself is legal (the built-in admin still exists), but it ends
    /// your own access — say so instead of letting it be a surprise.
    private func deleteMessage(for user: ManagedUser) -> String {
        let base = "Delete the account “\(user.username)”? They'll lose access immediately."
        guard isSelf(user) else { return base }
        return "Delete your own account “\(user.username)”? You'll be signed out and lose access immediately. The built-in admin can still sign in."
    }

    private func isSelf(_ user: ManagedUser) -> Bool {
        session.username == user.username
    }

    // MARK: List

    private var list: some View {
        List {
            Section {
                ForEach(users) { user in
                    NavigationLink {
                        UserDetailView(user: user, isSelf: isSelf(user)) { updated in
                            if let index = users.firstIndex(where: { $0.id == updated.id }) {
                                users[index] = updated
                            }
                        }
                    } label: {
                        row(user)
                    }
                    .listRowBackground(Theme.surface)
                }
                .onDelete(perform: confirmDelete)
            } footer: {
                Text("The built-in **admin** account is controlled by the server's ADMIN_PASSWORD and isn't listed here. Admins get full access; viewers get live view, events, recordings and the shared camera groups. Swipe a row to delete an account.")
            }
        }
        .listStyle(.plain)
        .scrollContentBackground(.hidden)
        .overlay {
            if loading && users.isEmpty {
                ProgressView().tint(Theme.accent)
            }
        }
    }

    private func row(_ user: ManagedUser) -> some View {
        HStack(spacing: 12) {
            Circle()
                .fill(user.role == .admin ? Theme.accent : Theme.textSecondary)
                .frame(width: 9, height: 9)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(user.username)
                        .font(.subheadline.weight(.medium))
                        .foregroundStyle(Theme.textPrimary)
                    if isSelf(user) {
                        Text("you")
                            .font(.caption2.weight(.semibold))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Theme.accent.opacity(0.18))
                            .foregroundStyle(Theme.accent)
                            .clipShape(Capsule())
                    }
                }
                Text(subtitle(user))
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
            }
            Spacer(minLength: 0)
        }
        .padding(.vertical, 4)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(user.username), \(user.role.rawValue)")
    }

    /// "Admin · created 12 Mar 2026" — role first, since it's the thing an admin
    /// scans this list for.
    private func subtitle(_ user: ManagedUser) -> String {
        let role = user.role.rawValue.capitalized
        let created = Date(timeIntervalSince1970: user.createdAt)
            .formatted(date: .abbreviated, time: .omitted)
        return "\(role) · created \(created)"
    }

    // MARK: Loading

    private func load() async {
        guard session.isAdmin, let api = session.api else { return }
        loading = users.isEmpty
        do {
            users = try await api.users()
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
        guard let index = offsets.first, users.indices.contains(index) else { return }
        pendingDelete = users[index]
    }

    private func delete(_ user: ManagedUser) async {
        guard let api = session.api else { return }
        busy = true
        defer { busy = false }
        pendingDelete = nil
        do {
            try await api.deleteUser(id: user.id)
            users.removeAll { $0.id == user.id }
            errorMessage = nil
        } catch {
            session.handleAPIError(error)
            errorMessage = "Delete failed: "
                + ((error as? ApiError)?.message ?? error.localizedDescription)
        }
    }
}

// MARK: - User detail (role + password reset)

/// One account's editor. Role changes commit immediately (optimistic, reverting
/// with the server's message — "Cannot demote the last admin" is the one that
/// actually bites). A password reset is an explicit action, since it's not
/// something to fire on a stray tap.
private struct UserDetailView: View {
    let isSelf: Bool
    /// Called with the server's version after a successful commit.
    let onChanged: (ManagedUser) -> Void

    @EnvironmentObject private var session: SessionModel

    @State private var user: ManagedUser
    @State private var newPassword = ""
    @State private var busy = false
    @State private var errorMessage: String?
    @State private var successMessage: String?

    init(user: ManagedUser, isSelf: Bool, onChanged: @escaping (ManagedUser) -> Void) {
        self.isSelf = isSelf
        self.onChanged = onChanged
        _user = State(initialValue: user)
    }

    private var canResetPassword: Bool {
        newPassword.count >= UsersView.minPasswordLength && !busy
    }

    private var roleBinding: Binding<Role> {
        Binding(
            get: { user.role },
            set: { role in
                guard role != user.role else { return }
                Task { await changeRole(to: role) }
            }
        )
    }

    var body: some View {
        Form {
            Section {
                Picker("Role", selection: roleBinding) {
                    Text("Viewer").tag(Role.viewer)
                    Text("Admin").tag(Role.admin)
                }
                .disabled(busy)
            } header: {
                Text("Role")
            } footer: {
                Text(roleFooter)
            }
            .listRowBackground(Theme.surface)

            Section {
                SecureField("New password", text: $newPassword)
                    .textContentType(.newPassword)
                Button {
                    Task { await resetPassword() }
                } label: {
                    HStack {
                        Text("Reset password")
                            .foregroundStyle(canResetPassword ? Theme.accent : Theme.textSecondary)
                        if busy {
                            Spacer()
                            ProgressView().tint(Theme.accent)
                        }
                    }
                }
                .disabled(!canResetPassword)
            } header: {
                Text("Password")
            } footer: {
                Text("At least \(UsersView.minPasswordLength) characters. This replaces \(user.username)'s password outright — there's no way to read the old one, so tell them the new one.")
            }
            .listRowBackground(Theme.surface)

            Section {
                LabeledContent("Username") {
                    Text(user.username).foregroundStyle(Theme.textPrimary)
                }
                LabeledContent("Created") {
                    Text(Date(timeIntervalSince1970: user.createdAt)
                            .formatted(date: .abbreviated, time: .shortened))
                        .foregroundStyle(Theme.textPrimary)
                }
            } footer: {
                Text("A username can't be changed after the account is created.")
            }
            .listRowBackground(Theme.surface)

            if let errorMessage {
                Section {
                    Text(errorMessage)
                        .font(.footnote)
                        .foregroundStyle(Theme.danger)
                }
                .listRowBackground(Theme.surface)
            } else if let successMessage {
                Section {
                    Text(successMessage)
                        .font(.footnote)
                        .foregroundStyle(Theme.success)
                }
                .listRowBackground(Theme.surface)
            }
        }
        .scrollContentBackground(.hidden)
        .background(Theme.bg)
        .navigationTitle(user.username)
        .navigationBarTitleDisplayMode(.inline)
    }

    private var roleFooter: String {
        let base = "Admins get full access, including cameras, users and system settings. Viewers get live view, events, recordings and the shared camera groups."
        guard isSelf else { return base }
        return base + " This is your own account — demoting yourself to viewer takes your admin access away immediately."
    }

    // MARK: Commits

    /// Optimistic, reverting on failure. The last-admin guard lives on the
    /// server; we surface its message rather than second-guessing it (we can't
    /// count admins reliably from one screen anyway).
    private func changeRole(to role: Role) async {
        guard let api = session.api else { return }
        let previous = user.role
        user.role = role
        busy = true
        defer { busy = false }
        do {
            let saved = try await api.updateUser(id: user.id, role: role)
            user = saved
            onChanged(saved)
            errorMessage = nil
            successMessage = "Role updated to \(saved.role.rawValue)."
        } catch {
            user.role = previous
            session.handleAPIError(error)
            successMessage = nil
            // 400 "Cannot demote the last admin" is the one worth reading.
            errorMessage = "Role change failed: "
                + ((error as? ApiError)?.message ?? error.localizedDescription)
        }
    }

    private func resetPassword() async {
        guard let api = session.api, canResetPassword else { return }
        busy = true
        defer { busy = false }
        do {
            let saved = try await api.updateUser(id: user.id, password: newPassword)
            user = saved
            onChanged(saved)
            newPassword = ""
            errorMessage = nil
            successMessage = "Password reset for \(saved.username)."
        } catch {
            session.handleAPIError(error)
            successMessage = nil
            errorMessage = "Password reset failed: "
                + ((error as? ApiError)?.message ?? error.localizedDescription)
        }
    }
}

// MARK: - Add sheet

/// POST /api/users — username, password and role. The hints mirror the backend's
/// validators so the common mistakes are caught before the round-trip, but the
/// server's `detail` is what's shown when one gets through.
private struct AddUserSheet: View {
    /// Called with the created user so the list behind us updates without a refetch.
    let onCreated: (ManagedUser) -> Void

    @EnvironmentObject private var session: SessionModel
    @Environment(\.dismiss) private var dismiss

    @State private var username = ""
    @State private var password = ""
    @State private var role: Role = .viewer
    @State private var saving = false
    @State private var errorMessage: String?

    private var trimmedUsername: String {
        username.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    }

    /// "admin" is reserved for the built-in env account (400 from the backend).
    private var isReserved: Bool { trimmedUsername == "admin" }

    private var usernameLooksValid: Bool {
        trimmedUsername.range(of: UsersView.usernamePattern, options: .regularExpression) != nil
    }

    private var canSave: Bool {
        usernameLooksValid && !isReserved && password.count >= UsersView.minPasswordLength
    }

    /// Live, specific hint — only once there's something to complain about.
    private var usernameProblem: String? {
        guard !trimmedUsername.isEmpty else { return nil }
        if isReserved { return "“admin” is reserved for the built-in account." }
        if !usernameLooksValid {
            return "3–32 characters: lowercase letters, digits, “_”, “.” or “-”, starting with a letter or digit."
        }
        return nil
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("jane", text: $username)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .textContentType(.username)
                        // Keep the name legal as it's typed rather than bouncing
                        // off the server's validator — same trick as the camera
                        // slug field in CamerasAdminView.
                        .onChange(of: username) { _, new in
                            let cleaned = new.lowercased().filter {
                                $0.isLowercase || $0.isNumber
                                    || $0 == "_" || $0 == "." || $0 == "-"
                            }
                            if cleaned != new { username = cleaned }
                        }
                } header: {
                    Text("Username")
                } footer: {
                    if let usernameProblem {
                        Text(usernameProblem).foregroundStyle(Theme.danger)
                    } else {
                        Text("3–32 characters: lowercase letters, digits, “_”, “.” or “-”. It can't be changed later.")
                    }
                }
                .listRowBackground(Theme.surface)

                Section {
                    SecureField("At least \(UsersView.minPasswordLength) characters", text: $password)
                        .textContentType(.newPassword)
                } header: {
                    Text("Password")
                } footer: {
                    Text("You're setting this for them — there's no invite email, so pass it on securely. They can't change it from the app yet.")
                }
                .listRowBackground(Theme.surface)

                Section {
                    Picker("Role", selection: $role) {
                        Text("Viewer").tag(Role.viewer)
                        Text("Admin").tag(Role.admin)
                    }
                } header: {
                    Text("Role")
                } footer: {
                    Text(role == .admin
                         ? "Admin — full access, including cameras, users and system settings."
                         : "Viewer — live view, events, recordings and the shared camera groups.")
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
            .navigationTitle("New User")
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

    private func save() async {
        guard let api = session.api, canSave, !saving else { return }
        saving = true
        defer { saving = false }
        do {
            let created = try await api.createUser(
                username: trimmedUsername, password: password, role: role
            )
            onCreated(created)
            dismiss()
        } catch {
            session.handleAPIError(error)
            // 409 "User 'x' already exists" / 400 reserved-or-invalid name.
            errorMessage = (error as? ApiError)?.message ?? error.localizedDescription
        }
    }
}
