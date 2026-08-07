/// Settings → Cameras & Detection → "Privacy Mode": the per-camera / per-group
/// capture kill switch (backend: app/privacy.py, GET/POST /api/privacy).
///
/// Turning a camera private stops ALL Vigilume capture for it — recording,
/// detection, events, notifications, live view, audio and on-camera-AI ingest —
/// while touching NOTHING on the camera itself. (The old hardware "lens mask"
/// that reconfigured the device is gone; this replaced it.)
///
/// **ADMIN-ONLY.** Both verbs of /api/privacy are `require_admin`, so this
/// screen must only ever be reachable from an admin-gated row — a viewer would
/// get a 403 and a dead screen. A viewer still SEES privacy state on the
/// dashboard, via `Camera.isPrivate` (any-authenticated) driving the tile
/// overlay; they simply cannot read the configuration or change it.
///
/// Applies IMMEDIATELY, deliberately — no Save button. Privacy is the switch
/// you reach for when you want capture to stop NOW; batching it behind Save
/// would leave a window where the UI claims private while cameras keep
/// recording. Mirrors the web PrivacyModeCard.
import SwiftUI

struct PrivacyModeView: View {
    @EnvironmentObject private var session: SessionModel

    @State private var state: PrivacyModeState?
    @State private var cameras: [Camera] = []
    @State private var groups: [CameraGroup] = []
    @State private var loadError: String?
    /// Non-nil while a toggle is in flight; also the key of the row to disable.
    @State private var busy: String?
    /// Optimistic overlay applied while a POST is in flight.
    ///
    /// Needed because `busy = key` mutates @State and re-renders the body
    /// BEFORE the response lands. Without this the binding's getter would still
    /// read the pre-tap `state`, so SwiftUI would drive the switch back OFF for
    /// the whole round trip (which awaits recorder/detector/stream teardown
    /// server-side) and then snap it ON — on a capture kill switch that reads
    /// as a malfunction. Cleared once the authoritative response is adopted.
    @State private var pendingPrivate: [String: Bool] = [:]
    @State private var pendingGroups: [Int: Bool] = [:]
    @State private var actionError: String?

    var body: some View {
        List {
            if let loadError {
                Section {
                    Text(loadError)
                        .font(.footnote)
                        .foregroundStyle(Theme.danger)
                }
                .listRowBackground(Theme.surface)
            }

            summarySection
            camerasSection
            if !groups.isEmpty { groupsSection }
        }
        .scrollContentBackground(.hidden)
        .background(Theme.bg)
        .navigationTitle("Privacy Mode")
        .navigationBarTitleDisplayMode(.inline)
        .task { await load() }
        .refreshable { await load() }
        .alert(
            "Privacy Mode",
            isPresented: Binding(
                get: { actionError != nil },
                set: { if !$0 { actionError = nil } }
            )
        ) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(actionError ?? "")
        }
    }

    // MARK: Sections

    private var summarySection: some View {
        Section {
            HStack {
                Image(systemName: privateCount == 0 ? "eye.fill" : "eye.slash.fill")
                    .foregroundStyle(privateCount == 0 ? Theme.success : Theme.warning)
                Text(summaryText)
                    .font(.subheadline)
                    .foregroundStyle(Theme.textPrimary)
            }
        } footer: {
            Text(
                "A camera in Privacy Mode stops recording, detecting and streaming "
                + "entirely. Nothing on the camera itself is changed."
            )
            .foregroundStyle(Theme.textSecondary)
        }
        .listRowBackground(Theme.surface)
    }

    private var camerasSection: some View {
        Section("Cameras") {
            ForEach(cameras) { cam in
                let viaGroup = isPrivateViaGroup(cam.name)
                let alsoDirect = state?.cameras.contains(cam.name) ?? false
                Toggle(isOn: cameraBinding(cam.name)) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(cam.friendlyName)
                            .foregroundStyle(Theme.textPrimary)
                        if viaGroup {
                            Text(alsoDirect
                                 ? "Selected here and private via its group"
                                 : "Private via its group")
                                .font(.caption2)
                                .foregroundStyle(Theme.textSecondary)
                        }
                    }
                }
                .tint(Theme.accent)
                // A camera made private by a GROUP can't be individually
                // un-privated — the group selection wins. Show it on and
                // locked rather than offering a toggle that would appear to
                // do nothing.
                .disabled(viaGroup || busy != nil)
            }
        }
        .listRowBackground(Theme.surface)
    }

    private var groupsSection: some View {
        Section("Groups") {
            ForEach(groups) { group in
                Toggle(isOn: groupBinding(group.id)) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(group.name)
                            .foregroundStyle(Theme.textPrimary)
                        Text(
                            group.cameras.isEmpty
                                ? "No cameras"
                                : "\(group.cameras.count) camera"
                                    + (group.cameras.count == 1 ? "" : "s")
                        )
                        .font(.caption2)
                        .foregroundStyle(Theme.textSecondary)
                    }
                }
                .tint(Theme.accent)
                .disabled(busy != nil)
            }
        }
        .listRowBackground(Theme.surface)
    }

    // MARK: Derived

    private var privateCount: Int { state?.privateCameras.count ?? 0 }

    private var summaryText: String {
        guard privateCount > 0 else { return "No cameras are private" }
        return "\(privateCount) camera\(privateCount == 1 ? "" : "s") private — not recording"
    }

    /// True when a SELECTED GROUP contains this camera — i.e. the group makes
    /// it private regardless of the direct selection.
    ///
    /// Computed from real group MEMBERSHIP, not from
    /// `privateCameras minus cameras`. That set-difference version was wrong
    /// for a camera selected BOTH directly and via a group: it reported "not
    /// via group", so the row stayed enabled with no caption, and turning it
    /// off silently dropped the direct selection while the camera visibly
    /// stayed private (the group still resolved it). The admin saw a toggle
    /// refuse them for no stated reason, and the dropped selection only
    /// surfaced later — when they deselected the group and that camera resumed
    /// capture along with the rest, despite having been picked individually.
    private func isPrivateViaGroup(_ name: String) -> Bool {
        guard let state else { return false }
        let selected = Set(state.groups)
        return groups.contains { selected.contains($0.id) && $0.cameras.contains(name) }
    }

    // MARK: Bindings (apply immediately)

    private func cameraBinding(_ name: String) -> Binding<Bool> {
        Binding(
            get: {
                if let pending = pendingPrivate[name] { return pending }
                return state?.privateCameras.contains(name) ?? false
            },
            set: { on in
                guard let state else { return }
                var next = Set(state.cameras)
                if on { next.insert(name) } else { next.remove(name) }
                pendingPrivate[name] = on
                Task { await apply(cameras: Array(next).sorted(), key: "cam:\(name)") }
            }
        )
    }

    private func groupBinding(_ id: Int) -> Binding<Bool> {
        Binding(
            get: {
                if let pending = pendingGroups[id] { return pending }
                return state?.groups.contains(id) ?? false
            },
            set: { on in
                guard let state else { return }
                var next = Set(state.groups)
                if on { next.insert(id) } else { next.remove(id) }
                pendingGroups[id] = on
                Task { await apply(groups: Array(next).sorted(), key: "grp:\(id)") }
            }
        )
    }

    // MARK: Networking

    private func load() async {
        // Defence in depth: the row that pushes this screen is already
        // admin-gated, but never issue an admin-only request as a viewer.
        guard session.isAdmin, let api = session.api else {
            loadError = "Privacy Mode is available to administrators only."
            return
        }
        do {
            async let p = api.privacyMode()
            async let c = api.cameras()
            async let g = api.groups()
            state = try await p
            cameras = try await c
            groups = try await g
            loadError = nil
        } catch {
            loadError = (error as? ApiError)?.message ?? error.localizedDescription
        }
    }

    private func apply(cameras camerasArg: [String]? = nil, groups groupsArg: [Int]? = nil,
                       key: String) async
    {
        guard session.isAdmin, let api = session.api, busy == nil else { return }
        busy = key
        defer { busy = nil }
        do {
            // Adopt the RESOLVED response rather than toggling locally: a
            // camera can become private through a group, and only the server
            // knows the effective set.
            state = try await api.setPrivacyMode(cameras: camerasArg, groups: groupsArg)
            // The authoritative resolved set has landed — drop the optimistic
            // overlay so the UI shows the server's answer, including a camera
            // that stayed private through a group.
            pendingPrivate.removeAll()
            pendingGroups.removeAll()
            // Camera rows carry `private`, which drives the dashboard overlay.
            self.cameras = try await api.cameras()
        } catch {
            actionError = (error as? ApiError)?.message ?? error.localizedDescription
            // Drop the optimistic overlay FIRST: a failed toggle must never
            // leave the UI claiming a privacy state the backend never accepted.
            pendingPrivate.removeAll()
            pendingGroups.removeAll()
            await load()
        }
    }
}
