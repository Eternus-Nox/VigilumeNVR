import SwiftUI

/// Settings › Cameras: the iOS twin of the web's Settings → Cameras tab
/// (frontend/src/pages/settings/CamerasTab.tsx) for the three things iOS had no
/// way to do — ADD, DELETE and REORDER. Editing is deliberately NOT duplicated
/// here: a row taps through to the existing `CameraSettingsView`, which already
/// owns the full per-camera config, credentials and device settings.
///
/// The order committed here is the global dashboard order (PUT
/// /api/cameras/order) — the same one the web drag-handle writes. Reorders apply
/// optimistically and revert if the server rejects them.
///
/// Admin-only: reached from `SettingsHomeView`'s `session.isAdmin` section, so
/// it lives inside that view's existing NavigationStack — no NavigationStack of
/// its own.
struct CamerasAdminView: View {
    @EnvironmentObject private var session: SessionModel

    @State private var cameras: [Camera] = []
    @State private var loading = true
    @State private var errorMessage: String?
    @State private var showingAdd = false
    /// Set while the delete confirmation for a camera is up.
    @State private var pendingDelete: Camera?
    @State private var busy = false

    /// The known Amcrest models the add form offers — mirrors KNOWN_MODELS in
    /// the web CamerasTab. Editing an existing camera (CameraSettingsView) keeps
    /// its own list, including the "other/unknown" escape hatch.
    static let knownModels = [
        "IP5M-T1277EW-AI", "IP8M-2779EW-AI", "AD410", "IP3M-941B", "IP4M-1041B", "IP4M-1056E",
    ]

    var body: some View {
        Group {
            if let errorMessage, cameras.isEmpty {
                ContentUnavailableView(
                    "Couldn't load cameras",
                    systemImage: "video.slash",
                    description: Text(errorMessage)
                )
            } else if cameras.isEmpty && !loading {
                ContentUnavailableView(
                    "No cameras",
                    systemImage: "video.slash",
                    description: Text("Tap + to add your first camera.")
                )
            } else {
                list
            }
        }
        .background(Theme.bg)
        .navigationTitle("Cameras")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button {
                    showingAdd = true
                } label: {
                    Image(systemName: "plus")
                }
                .accessibilityLabel("Add camera")
                .disabled(busy)
            }
            if !cameras.isEmpty {
                ToolbarItem(placement: .topBarTrailing) {
                    EditButton()
                }
            }
        }
        .sheet(isPresented: $showingAdd) {
            AddCameraSheet { await load() }
        }
        // Deleting a camera is destructive enough to name it in the prompt —
        // matches the web's ConfirmDialog rather than a bare swipe-to-delete.
        .alert(
            "Remove camera",
            isPresented: Binding(
                get: { pendingDelete != nil },
                set: { if !$0 { pendingDelete = nil } }
            ),
            presenting: pendingDelete
        ) { camera in
            Button("Remove", role: .destructive) {
                Task { await delete(camera) }
            }
            Button("Cancel", role: .cancel) { pendingDelete = nil }
        } message: { camera in
            Text("Remove \(displayName(camera))? Its recordings and events are not deleted.")
        }
        .task { await load() }
        .refreshable { await load() }
    }

    // MARK: List

    private var list: some View {
        List {
            Section {
                ForEach(cameras) { camera in
                    NavigationLink {
                        CameraSettingsView(camera: camera)
                    } label: {
                        row(camera)
                    }
                    .listRowBackground(Theme.surface)
                }
                .onMove(perform: move)
                .onDelete(perform: confirmDelete)
            } footer: {
                if cameras.count > 1 {
                    Text("Tap a camera to edit it. Tap Edit to drag cameras into the dashboard order, or swipe a row to remove it.")
                } else {
                    Text("Tap a camera to edit it. Swipe a row to remove it.")
                }
            }
        }
        .listStyle(.plain)
        .scrollContentBackground(.hidden)
        .overlay {
            if loading && cameras.isEmpty {
                ProgressView().tint(Theme.accent)
            }
        }
    }

    private func row(_ camera: Camera) -> some View {
        HStack(spacing: 12) {
            // Live status beats the snapshot's `online` when the socket has a
            // fresher value (same source the rest of the app trusts).
            Circle()
                .fill(isOnline(camera) ? Theme.success : Theme.danger)
                .frame(width: 9, height: 9)
                .accessibilityLabel(isOnline(camera) ? "Online" : "Offline")
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(displayName(camera))
                        .font(.subheadline.weight(.medium))
                        .foregroundStyle(Theme.textPrimary)
                    if camera.needsCredentials {
                        Text("needs password")
                            .font(.caption2.weight(.semibold))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Theme.warning.opacity(0.18))
                            .foregroundStyle(Theme.warning)
                            .clipShape(Capsule())
                    }
                }
                Text("\(camera.model) · \(camera.ip)")
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
            }
            Spacer(minLength: 0)
        }
        .padding(.vertical, 4)
    }

    private func isOnline(_ camera: Camera) -> Bool {
        session.cameraOnline[camera.name] ?? camera.online
    }

    /// Friendly name, falling back to the prettified slug (front_yard → Front
    /// Yard) — same fallback as the web's `titleCase(cam.name)`.
    private func displayName(_ camera: Camera) -> String {
        camera.friendlyName.isEmpty
            ? camera.name.replacingOccurrences(of: "_", with: " ").capitalized
            : camera.friendlyName
    }

    // MARK: Loading

    private func load() async {
        guard let api = session.api else { return }
        loading = cameras.isEmpty
        do {
            cameras = try await api.cameras()
            errorMessage = nil
        } catch {
            session.handleAPIError(error)
            errorMessage = (error as? ApiError)?.message ?? error.localizedDescription
        }
        loading = false
    }

    // MARK: Reorder

    /// Apply the drag optimistically, then commit. A failed commit puts the old
    /// order back rather than leaving the list lying about the dashboard.
    private func move(from source: IndexSet, to destination: Int) {
        guard let api = session.api else { return }
        let previous = cameras
        cameras.move(fromOffsets: source, toOffset: destination)
        let names = cameras.map(\.name)
        Task {
            busy = true
            defer { busy = false }
            do {
                try await api.setCameraOrder(names)
                errorMessage = nil
            } catch {
                cameras = previous
                session.handleAPIError(error)
                errorMessage = "Reorder failed: "
                    + ((error as? ApiError)?.message ?? error.localizedDescription)
            }
        }
    }

    // MARK: Delete

    /// Swipe/Edit-mode delete only ARMS the confirmation — the row stays until
    /// the alert is confirmed.
    private func confirmDelete(at offsets: IndexSet) {
        guard let index = offsets.first, cameras.indices.contains(index) else { return }
        pendingDelete = cameras[index]
    }

    private func delete(_ camera: Camera) async {
        guard let api = session.api else { return }
        busy = true
        defer { busy = false }
        pendingDelete = nil
        do {
            try await api.deleteCamera(name: camera.name)
            cameras.removeAll { $0.name == camera.name }
            errorMessage = nil
        } catch {
            session.handleAPIError(error)
            errorMessage = "Delete failed: "
                + ((error as? ApiError)?.message ?? error.localizedDescription)
        }
    }
}

// MARK: - Add sheet

/// POST /api/cameras form — the add half of the web's camera modal. Only the
/// fields the create contract needs: everything else (detect objects/mode,
/// exempt zones, stream overrides, device settings) is edited afterwards in
/// CameraSettingsView, once the backend has probed the camera's capabilities.
private struct AddCameraSheet: View {
    /// Called after a successful add so the list refetches before the sheet goes.
    let onAdded: () async -> Void

    @EnvironmentObject private var session: SessionModel
    @Environment(\.dismiss) private var dismiss

    @State private var name = ""
    @State private var friendlyName = ""
    // Auto-detect by default: the backend asks the camera its own model
    // (getDeviceType) on save and adopts it. Defaulting to a concrete model
    // would ASSERT that model and suppress detection entirely.
    @State private var model = "unknown"
    @State private var ip = ""
    @State private var username = "admin"
    @State private var password = ""
    @State private var saving = false
    @State private var errorMessage: String?

    /// The backend enforces `^[a-z][a-z0-9_]{0,31}$` on the name and validates
    /// the IP + model; we only gate on "the required fields are filled" and let
    /// the server's readable detail carry the rest.
    private var canSave: Bool {
        !name.trimmed.isEmpty && !ip.trimmed.isEmpty
            && !username.trimmed.isEmpty && !password.isEmpty
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("front_yard", text: $name)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        // Keep the slug legal as it's typed rather than
                        // bouncing off the server's validator.
                        .onChange(of: name) { _, new in
                            let cleaned = new.lowercased().filter {
                                $0.isLowercase || $0.isNumber || $0 == "_"
                            }
                            if cleaned != new { name = cleaned }
                        }
                    TextField("Front Yard", text: $friendlyName)
                    Picker("Model", selection: $model) {
                        // Selecting this IS the auto-detect action: the backend
                        // probes getDeviceType on save and adopts the match.
                        Text("Auto-detect (recommended)").tag("unknown")
                        ForEach(CamerasAdminView.knownModels, id: \.self) { known in
                            Text(known).tag(known)
                        }
                    }
                    TextField("192.168.1.101", text: $ip)
                        .keyboardType(.numbersAndPunctuation)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                } header: {
                    Text("Camera")
                } footer: {
                    Text("Name is the URL-safe slug used for streams and recordings — lowercase letters, digits and underscores. It can't be changed later.")
                }
                .listRowBackground(Theme.surface)

                Section {
                    TextField("admin", text: $username)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                    SecureField("Password", text: $password)
                } header: {
                    Text("Camera credentials")
                } footer: {
                    Text("The camera's own username and password — needed for streaming, IR, siren, doorbell alerts and capability detection.")
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
            .navigationTitle("Add Camera")
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
                        Button("Save") { Task { await save() } }
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
        let trimmedName = name.trimmed
        let payload = CameraCreatePayload(
            name: trimmedName,
            // An empty friendly name would leave the list showing the raw slug;
            // prettify it the same way the row fallback does.
            friendlyName: friendlyName.trimmed.isEmpty
                ? trimmedName.replacingOccurrences(of: "_", with: " ").capitalized
                : friendlyName.trimmed,
            model: model,
            ip: ip.trimmed,
            username: username.trimmed,
            password: password
            // detectObjects / detectFps omitted — the backend applies its
            // defaults, and both are editable afterwards in camera settings.
        )
        do {
            try await api.addCamera(payload)
            await onAdded()
            dismiss()
        } catch {
            session.handleAPIError(error)
            errorMessage = (error as? ApiError)?.message ?? error.localizedDescription
        }
    }
}

private extension String {
    var trimmed: String { trimmingCharacters(in: .whitespacesAndNewlines) }
}
