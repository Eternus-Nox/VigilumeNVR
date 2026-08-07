import SwiftUI

/// Cameras tab — the live dashboard. A single-column vertical list of
/// full-width 16:9 muted HLS tiles (bigger view per camera), group filter
/// chips (All + shared groups from the API), live online/offline badges from
/// WS camera_status, pull-to-refresh. Tapping a tile opens CameraDetailView —
/// the one-camera screen with the big live player AND all controls visible
/// (long-press only offers a "full-screen live" shortcut). LazyVStack keeps
/// attach/detach lazy: only on-screen tiles hold a stream.
struct CamerasView: View {
    @EnvironmentObject private var session: SessionModel

    @State private var cameras: [Camera] = []
    @State private var groups: [CameraGroup] = []
    @State private var selectedGroupID: Int?
    @State private var fullscreenCamera: Camera?
    @State private var detailCamera: Camera?
    @State private var errorMessage: String?
    @State private var isLoading = false
    @State private var hasLoaded = false

    var body: some View {
        NavigationStack {
            Group {
                if let errorMessage, cameras.isEmpty {
                    ContentUnavailableView(
                        "Couldn't load cameras",
                        systemImage: "video.slash",
                        description: Text(errorMessage)
                    )
                } else if cameras.isEmpty, !hasLoaded {
                    ProgressView()
                        .tint(Theme.accent)
                } else if cameras.isEmpty {
                    ContentUnavailableView(
                        "No cameras",
                        systemImage: "video.slash",
                        description: Text("Add cameras from the web dashboard")
                    )
                } else {
                    cameraList
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Theme.bg)
            .navigationTitle("Cameras")
            .task { await load() }
            .onReceive(session.wsReconnected) { _ in
                Task { await load() }
            }
            // Privacy Mode (and any other camera-row change) arrives ONLY as
            // this message. It is not a status change — a private camera stays
            // "online", because privacy is a software gate that never touches
            // the camera. Without this the grid keeps rendering live tiles for a
            // camera the backend has already stopped capturing.
            .onReceive(session.wsMessages) { message in
                if case .camerasChanged = message { Task { await load() } }
            }
            // Answered a doorbell call: open that camera's full-screen live view.
            .onChange(of: session.pendingLiveCameraName) { _, name in
                consumePendingLive(name)
            }
            .onAppear { consumePendingLive(session.pendingLiveCameraName) }
            .fullScreenCover(item: $fullscreenCamera) { camera in
                SingleCameraView(
                    camera: camera,
                    whepURL: session.api?.liveSubStreamWHEPURL(camera: camera.name),
                    whepHighURL: session.api?.liveStreamWHEPURL(camera: camera.name),
                    primaryURL: session.api?.liveStreamURL(camera: camera.name),
                    fallbackURL: session.api?.liveSubStreamURL(camera: camera.name)
                )
            }
            .navigationDestination(item: $detailCamera) { camera in
                CameraDetailView(camera: camera)
            }
        }
    }

    // MARK: Camera list (single column — one full-width tile per camera)

    private var cameraList: some View {
        // Outer reader gives the scroll viewport height; the named coordinate
        // space lets each tile tell whether it's actually on screen, so only
        // visible tiles ever hold a live stream (docs/ios-design.md §2).
        GeometryReader { outer in
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    if !groups.isEmpty {
                        groupChips
                    }
                    LazyVStack(spacing: 12) {
                        ForEach(visibleCameras) { camera in
                            CameraTileView(
                                camera: camera,
                                streamURL: session.api?.liveSubStreamURL(camera: camera.name),
                                fallbackURL: session.api?.liveStreamURL(camera: camera.name),
                                whepURL: session.api?.liveSubStreamWHEPURL(camera: camera.name),
                                posterURL: session.api?.cameraSnapshotURL(camera.name),
                                isOnline: isOnline(camera),
                                viewportHeight: outer.size.height,
                                onTap: { detailCamera = camera }
                            )
                            // Long-press: OPTIONAL shortcuts only — the plain
                            // tap already lands on the player + every control.
                            .contextMenu {
                                Button {
                                    detailCamera = camera
                                } label: {
                                    Label("Open camera", systemImage: "video")
                                }
                                Button {
                                    fullscreenCamera = camera
                                } label: {
                                    Label("Full-screen live", systemImage: "arrow.up.left.and.arrow.down.right")
                                }
                            }
                        }
                    }
                }
                .padding(.horizontal, 12)
                .padding(.top, 4)
                .padding(.bottom, 16)
            }
            .coordinateSpace(name: LiveVisibility.coordinateSpace)
            .scrollIndicators(.hidden)
            .refreshable { await load() }
        }
    }

    // MARK: Group filter chips

    private var groupChips: some View {
        ScrollView(.horizontal) {
            HStack(spacing: 8) {
                chip("All", id: nil)
                ForEach(groups) { group in
                    chip(group.name, id: group.id)
                }
            }
        }
        .scrollIndicators(.hidden)
    }

    private func chip(_ label: String, id: Int?) -> some View {
        let isSelected = selectedGroupID == id
        return Button {
            selectedGroupID = id
        } label: {
            Text(label)
                .font(.subheadline.weight(.medium))
                .foregroundStyle(isSelected ? Color.black : Theme.textPrimary)
                .padding(.horizontal, 14)
                .padding(.vertical, 7)
                .background(Capsule().fill(isSelected ? Theme.accent : Theme.surface))
                .overlay(
                    Capsule().stroke(isSelected ? Color.clear : Theme.border, lineWidth: 1)
                )
        }
        .buttonStyle(.plain)
    }

    // MARK: Data

    /// Group order is authoritative for its tiles; unknown names are skipped
    /// (the API tolerates stale names in groups — so do we).
    private var visibleCameras: [Camera] {
        guard let selectedGroupID,
              let group = groups.first(where: { $0.id == selectedGroupID })
        else { return cameras }
        return group.cameras.compactMap { name in
            cameras.first { $0.name == name }
        }
    }

    /// WS camera_status overrides the fetched snapshot when present.
    private func isOnline(_ camera: Camera) -> Bool {
        session.cameraOnline[camera.name] ?? camera.online
    }

    /// Open the full-screen live view for an answered doorbell call. The push
    /// payload's `camera` is the friendly name (e.g. "Front Door"), so match on
    /// friendlyName first, then the internal name (case-insensitive) to be
    /// robust to either. If the list hasn't loaded yet, keep the pending value
    /// and retry after `load()`; give up (clear it) only once loaded with no
    /// match, so a stale name can't wedge the tab.
    private func consumePendingLive(_ name: String?) {
        guard let name else { return }
        let match = cameras.first {
            $0.friendlyName.caseInsensitiveCompare(name) == .orderedSame
                || $0.name.caseInsensitiveCompare(name) == .orderedSame
        }
        if let match {
            fullscreenCamera = match
            session.pendingLiveCameraName = nil
        } else if hasLoaded {
            session.pendingLiveCameraName = nil
        }
    }

    private func load() async {
        guard let api = session.api else { return }
        isLoading = true
        defer {
            isLoading = false
            hasLoaded = true
        }
        do {
            cameras = try await api.cameras()
            errorMessage = nil
            // Re-point any OPEN detail/fullscreen presentation at the fresh row.
            // Both are driven by a @State Camera VALUE, so without this they
            // keep whatever was true when they opened — and the moment that
            // matters most for Privacy Mode is when someone enables it while you
            // are watching that camera. `Camera.id` is its name, so the id is
            // unchanged and SwiftUI re-renders in place rather than dismissing.
            if let open = fullscreenCamera,
               let fresh = cameras.first(where: { $0.name == open.name }) {
                fullscreenCamera = fresh
            }
            if let open = detailCamera,
               let fresh = cameras.first(where: { $0.name == open.name }) {
                detailCamera = fresh
            }
            // A doorbell call may have been answered before the list loaded —
            // now that we have cameras, open the matching live view.
            consumePendingLive(session.pendingLiveCameraName)
        } catch {
            session.handleAPIError(error)
            if cameras.isEmpty {
                errorMessage = (error as? ApiError)?.message ?? error.localizedDescription
            }
        }
        // Groups are a filter nicety — a failure never blocks the list.
        if let fetched = try? await api.groups() {
            groups = fetched.sorted { $0.position < $1.position }
            if let selected = selectedGroupID,
               !groups.contains(where: { $0.id == selected }) {
                selectedGroupID = nil
            }
        }
    }
}
