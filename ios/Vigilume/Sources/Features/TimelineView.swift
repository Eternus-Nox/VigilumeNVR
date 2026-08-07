import SwiftUI

/// Timeline tab — 24/7 continuous-recording review as a MULTI-CAMERA SYNCED
/// player, mirroring the web client's Timeline page in native SwiftUI:
/// - two-tier camera selection: a selector menu (All / a saved group / a custom
///   set) picks the AGGREGATE set that feeds the shared bar's union coverage; a
///   multi-select on-view chip row toggles WHICH of those cameras get a live
///   tile (capped at MAX_SYNC_PLAYERS),
/// - date navigation (‹ day › / Today),
/// - a SyncGridView of one AVPlayer tile per on-view camera (1 -> full width,
///   2+ -> a 2-column grid); a camera with no footage at the playhead shows a
///   placeholder while the others keep playing,
/// - ONE unified scrub bar (TimelineBarView) drawing union coverage across the
///   selected cameras + a draggable playhead; scrubbing it seeks EVERY tile to
///   the same wall-clock instant,
/// - ONE shared transport (TimelineTransportView) — play/pause, ±10 s, speed,
///   mute, jump-to-newest — driving the whole group via TimelineSyncCoordinator,
/// - range export: drag-select a span -> share/download the export.mp4 URL per
///   chosen on-view camera.
struct TimelineView: View {
    @EnvironmentObject private var session: SessionModel

    static let maxSyncPlayers = 4

    @StateObject private var coordinator = TimelineSyncCoordinator()

    @State private var recCameras: [RecordingCamera]?
    @State private var groups: [CameraGroup] = []
    /// Seeded from the persisted pick and re-resolved once the camera list
    /// lands (loadCatalog). It CANNOT be defaulted to the first camera here:
    /// at property-initialiser time recCameras is nil, so no camera name is
    /// knowable yet. `.all` is only the placeholder for that gap — the real
    /// default is applied in loadCatalog.
    @State private var selection: TimelineSelection = TimelineSelectionStore.load() ?? .all
    @State private var date = Date()
    @State private var dateAnchored = false
    @State private var dayData: [String: CameraDayData] = [:]
    /// Ordered subset of selectedCameras with a live tile (capped at 4).
    @State private var onView: [String] = []
    /// On-view cameras chosen for range export.
    @State private var exportCams: Set<String> = []
    @State private var zoom: TimelineZoom = .day
    @State private var rangeMode = false
    @State private var exportRange: ClosedRange<Double>?
    @State private var loading = false
    @State private var errorMessage: String?
    @State private var showCustomPicker = false
    @State private var requestSeq = 0
    /// Set for exactly one selection assignment: the computed default applied
    /// by loadCatalog. Keeps that write out of the persistence store so the
    /// "never chosen" sentinel survives.
    @State private var applyingDefault = false

    var body: some View {
        NavigationStack {
            Group {
                if let errorMessage, recCameras == nil {
                    ContentUnavailableView(
                        "Couldn't load recordings",
                        systemImage: "clock.badge.xmark",
                        description: Text(errorMessage)
                    )
                } else if let recCameras, recCameras.isEmpty {
                    ContentUnavailableView(
                        "No recordings",
                        systemImage: "clock",
                        description: Text("24/7 recordings will appear here")
                    )
                } else if recCameras == nil {
                    ProgressView().tint(Theme.accent)
                } else {
                    timelineContent
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Theme.bg)
            .navigationTitle("Timeline")
            .navigationBarTitleDisplayMode(.inline)
            .task { await loadCatalog() }
            // The timeline's AVPlayers need an ACTIVE playback session to be
            // audible. It never claimed one — it just inherited whatever was
            // active — so once live view started releasing the session on
            // teardown, scrubbed footage played silent. Claim one here (and
            // release it in onDisappear; RTCAudioSession reference-counts).
            .onAppear { LiveAudioSession.activatePlayback() }
            .onChange(of: selection) { _, newValue in
                // Persist HERE rather than at each of the four assignment sites
                // (menu / group / custom / the picker sheet's binding), so a
                // future fifth site cannot silently skip saving.
                //
                // ...but NOT when loadCatalog just applied the computed default.
                // Persisting that would write a pick the user never made, and
                // the store's whole design rests on `nil` meaning "never chosen"
                // — collapse it and the default can never be recomputed, so a
                // camera added at position 0 (or the current one's footage
                // ageing out) could never move the default again.
                if applyingDefault {
                    applyingDefault = false
                } else {
                    TimelineSelectionStore.save(newValue)
                }
                exportRange = nil
                syncOnView()
                anchorDateToSelection()
                Task { await loadDay() }
            }
            .onChange(of: date) { _, _ in
                exportRange = nil
                Task { await loadDay() }
            }
            .onChange(of: onView) { _, _ in reconcileOnViewTiles() }
            .onDisappear {
                coordinator.pause()
                LiveAudioSession.deactivate()   // balance the onAppear claim
            }
            .sheet(isPresented: $showCustomPicker) {
                CustomCameraPickerSheet(
                    cameras: recCameras ?? [],
                    selection: $selection
                )
                .presentationDetents([.medium, .large])
            }
        }
    }

    // MARK: Derived state

    private var selectedCameras: [String] {
        guard let recCameras else { return [] }
        return Self.cameras(for: selection, cams: recCameras, groups: groups)
    }

    /// Resolve a selection to camera names. Static so loadCatalog can ask
    /// "would this selection show anything?" before committing to it.
    static func cameras(
        for selection: TimelineSelection, cams: [RecordingCamera], groups: [CameraGroup]
    ) -> [String] {
        let known = cams.map(\.camera)
        switch selection {
        case .all:
            return known
        case .group(let id):
            guard let group = groups.first(where: { $0.id == id }) else { return [] }
            return group.cameras.filter { known.contains($0) }
        case .custom(let names):
            return known.filter { names.contains($0) }
        }
    }

    private var dayStart: Double { TimelineTime.dayStart(of: date) }
    private var dayEnd: Double { dayStart + TimelineTime.day }

    private var barWindow: (start: Double, end: Double) {
        switch zoom {
        case .day: return (dayStart, dayEnd)
        case .hour: return TimelineTime.hourWindow(containing: coordinator.playhead, dayStart: dayStart, dayEnd: dayEnd)
        }
    }

    /// Event marks across the whole selected set, merged onto the one shared bar
    /// (it draws a union, not per-camera lanes).
    private var eventTimes: [Double] {
        selectedCameras.flatMap { dayData[$0]?.eventTimes ?? [] }
    }

    private var coverage: [[RecordingRange]] {
        selectedCameras.map { dayData[$0]?.index?.ranges ?? [] }
    }

    private var hasAnyCoverage: Bool {
        coverage.contains { !$0.isEmpty }
    }

    private var newestCoverageEnd: Double? {
        coverage.compactMap { $0.last?.end }.max()
    }

    private func friendlyName(for camera: String) -> String {
        recCameras?.first(where: { $0.camera == camera })
            .map { $0.friendlyName.isEmpty ? camera : $0.friendlyName }
            ?? camera.replacingOccurrences(of: "_", with: " ").capitalized
    }

    private func hasFootage(_ camera: String) -> Bool {
        !(dayData[camera]?.index?.ranges.isEmpty ?? true)
    }

    private var selectionTitle: String {
        switch selection {
        case .all: return "All cameras"
        case .group(let id): return groups.first(where: { $0.id == id })?.name ?? "Group"
        case .custom(let names):
            // One camera is now the DEFAULT, not an exotic custom set, so
            // "Custom (1)" would be the first thing every user sees. Name it.
            if names.count == 1, let only = names.first { return friendlyName(for: only) }
            return "Custom (\(names.count))"
        }
    }

    /// The selection to open with: the user's persisted pick when it still
    /// resolves to something, otherwise the first camera.
    ///
    /// "First" is the same camera the dashboard shows first — the API returns
    /// cameras ORDER BY position, name and neither client re-sorts. Preference
    /// goes to the first camera that HAS footage: defaulting onto one that has
    /// never recorded opens the timeline on a permanently empty bar, which
    /// reads as a broken screen rather than an empty one.
    static func resolveSelection(
        _ stored: TimelineSelection?, for cams: [RecordingCamera]
    ) -> TimelineSelection {
        guard !cams.isEmpty else { return stored ?? .all }
        if let stored {
            // A stored pick can go stale — a renamed or deleted camera, a
            // deleted group. Falling through to the default beats showing an
            // empty timeline the user cannot explain.
            switch stored {
            case .all:
                return .all
            case .group:
                return stored          // membership is resolved later, against groups
            case .custom(let names):
                let known = Set(cams.map(\.camera))
                if !names.intersection(known).isEmpty { return stored }
            }
        }
        let first = cams.first(where: \.hasRecordings) ?? cams[0]
        return .custom([first.camera])
    }

    // MARK: Layout

    private var timelineContent: some View {
        ScrollView {
            VStack(spacing: 12) {
                controlsRow
                SyncGridView(models: coordinator.models, onRemove: { removeFromView($0) })
                readoutRow
                if selectedCameras.count > 1 { onViewChipRow }
                TimelineBarView(
                    viewStart: barWindow.start,
                    viewEnd: barWindow.end,
                    coverage: coverage,
                    eventTimes: eventTimes,
                    playhead: coordinator.playhead,
                    rangeMode: rangeMode,
                    range: $exportRange,
                    onScrubStart: { coordinator.isScrubbing = true },
                    onScrub: { coordinator.scrub(to: TimelineTime.clamp($0, dayStart, dayEnd - 1)) },
                    onScrubEnd: { t in
                        coordinator.seek(toWall: TimelineTime.clamp(t, dayStart, dayEnd - 1))
                    }
                )
                .padding(.horizontal, 16)
                hourNavRow
                TimelineTransportView(coordinator: coordinator, disabled: !hasAnyCoverage)
                if rangeMode { rangeExportBar }
                if !loading && !hasAnyCoverage && !selectedCameras.isEmpty {
                    Text("No continuous recording saved for the selected cameras on this day.")
                        .font(.caption)
                        .foregroundStyle(Theme.textSecondary)
                        .padding(.top, 4)
                }
                if selectedCameras.isEmpty {
                    Text("No cameras selected — pick a group or choose cameras above.")
                        .font(.caption)
                        .foregroundStyle(Theme.textSecondary)
                        .padding(.top, 4)
                }
            }
            .padding(.vertical, 12)
        }
        .scrollIndicators(.hidden)
    }

    private var controlsRow: some View {
        VStack(spacing: 10) {
            HStack(spacing: 10) {
                cameraSelectorMenu
                Spacer(minLength: 8)
                rangeButton
                zoomToggle
            }
            dateNav
        }
        .padding(.horizontal, 16)
    }

    private var cameraSelectorMenu: some View {
        Menu {
            Button {
                selection = .all
            } label: {
                Label("All cameras", systemImage: selection == .all ? "checkmark" : "video")
            }
            if !groups.isEmpty {
                Section("Groups") {
                    ForEach(groups) { group in
                        Button {
                            selection = .group(group.id)
                        } label: {
                            if selection == .group(group.id) {
                                Label(group.name, systemImage: "checkmark")
                            } else {
                                Text(group.name)
                            }
                        }
                    }
                }
            }
            Button {
                if case .custom = selection {} else {
                    selection = .custom(Set(selectedCameras))
                }
                showCustomPicker = true
            } label: {
                Label("Custom…", systemImage: "checklist")
            }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "video.fill").font(.caption)
                Text(selectionTitle).font(.subheadline.weight(.medium))
                Image(systemName: "chevron.down").font(.caption2)
            }
            .foregroundStyle(Theme.textPrimary)
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(Capsule().fill(Theme.surface))
            .overlay(Capsule().stroke(Theme.border, lineWidth: 1))
        }
    }

    private var zoomToggle: some View {
        HStack(spacing: 0) {
            zoomButton("Day", .day)
            zoomButton("1 h", .hour)
        }
        .background(Capsule().fill(Theme.surface))
        .overlay(Capsule().stroke(Theme.border, lineWidth: 1))
        .clipShape(Capsule())
    }

    private func zoomButton(_ title: String, _ value: TimelineZoom) -> some View {
        Button {
            zoom = value
        } label: {
            Text(title)
                .font(.footnote.weight(zoom == value ? .semibold : .regular))
                .foregroundStyle(zoom == value ? Theme.bgDeep : Theme.textPrimary)
                .padding(.horizontal, 12)
                .padding(.vertical, 7)
                .background(zoom == value ? Theme.accent : .clear)
        }
        .buttonStyle(.plain)
    }

    private var dateNav: some View {
        HStack(spacing: 10) {
            Button {
                date = TimelineTime.shift(date, byDays: -1)
            } label: {
                Image(systemName: "chevron.left")
                    .frame(width: 34, height: 30)
            }
            .buttonStyle(.bordered)
            .tint(Theme.accent)

            Text(date.formatted(.dateTime.weekday(.abbreviated).month(.abbreviated).day()))
                .font(.subheadline.weight(.medium))
                .foregroundStyle(Theme.textPrimary)
                .frame(minWidth: 110)

            Button {
                date = TimelineTime.shift(date, byDays: 1)
            } label: {
                Image(systemName: "chevron.right")
                    .frame(width: 34, height: 30)
            }
            .buttonStyle(.bordered)
            .tint(Theme.accent)
            .disabled(TimelineTime.isToday(date))

            Button("Today") {
                date = Date()
            }
            .font(.footnote.weight(.medium))
            .buttonStyle(.bordered)
            .tint(Theme.accent)
            .disabled(TimelineTime.isToday(date))

            Spacer()
        }
    }

    /// Compact one-line range-mode toggle (icon + "Range").
    private var rangeButton: some View {
        Button {
            rangeMode.toggle()
            if !rangeMode { exportRange = nil }
        } label: {
            HStack(spacing: 5) {
                Image(systemName: "selection.pin.in.out")
                    .font(.caption)
                Text("Range")
                    .font(.footnote.weight(.semibold))
            }
            .lineLimit(1)
            .fixedSize(horizontal: true, vertical: false)
            .foregroundStyle(rangeMode ? Theme.bgDeep : Theme.textPrimary)
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(Capsule().fill(rangeMode ? Theme.warning : Theme.surface))
            .overlay(Capsule().stroke(rangeMode ? Theme.warning : Theme.border, lineWidth: 1))
        }
        .buttonStyle(.plain)
        .disabled(selectedCameras.isEmpty)
        .accessibilityLabel(rangeMode ? "Exit range selection" : "Select export range")
    }

    /// Time readout under the grid, labelled with the leader camera — the
    /// footage currently driving the shared playhead.
    private var readoutRow: some View {
        HStack {
            Text(TimelineTime.clockLabel(coordinator.playhead))
                .font(.subheadline.monospacedDigit().weight(.semibold))
                .foregroundStyle(Theme.textPrimary)
            if let leader = coordinator.leaderCamera {
                Text("· \(friendlyName(for: leader))")
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
            }
            Spacer()
        }
        .padding(.horizontal, 16)
    }

    /// Prev/next hour buttons (only meaningful in the 1 h zoom).
    @ViewBuilder
    private var hourNavRow: some View {
        if zoom == .hour {
            HStack {
                Button {
                    coordinator.seek(toWall: max(dayStart, barWindow.start - TimelineTime.hour))
                } label: {
                    Label("Prev hour", systemImage: "chevron.left.2").font(.caption)
                }
                .buttonStyle(.bordered)
                .tint(Theme.accent)
                .disabled(barWindow.start <= dayStart)
                Spacer()
                Button {
                    coordinator.seek(toWall: min(dayEnd - 1, barWindow.start + TimelineTime.hour))
                } label: {
                    Label("Next hour", systemImage: "chevron.right.2").font(.caption)
                }
                .buttonStyle(.bordered)
                .tint(Theme.accent)
                .disabled(barWindow.end >= dayEnd)
            }
            .padding(.horizontal, 16)
        }
    }

    /// Multi-select chip row: tap to toggle whether a selected camera has a live
    /// tile in the grid (capped at MAX_SYNC_PLAYERS). Shows on/off + "no footage".
    private var onViewChipRow: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(selectedCameras, id: \.self) { name in
                    let on = onView.contains(name)
                    let footage = hasFootage(name)
                    Button {
                        toggleView(name)
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: on ? "checkmark.circle.fill" : "circle")
                                .font(.caption2)
                            Text(friendlyName(for: name))
                                .font(.footnote.weight(on ? .semibold : .regular))
                            if !footage {
                                Text("· no footage")
                                    .font(.caption2)
                                    .foregroundStyle(on ? Theme.bgDeep.opacity(0.7) : Theme.textSecondary)
                            }
                        }
                        .foregroundStyle(on ? Theme.bgDeep : Theme.textPrimary)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(Capsule().fill(on ? Theme.accent : Theme.surface))
                        .overlay(Capsule().stroke(on ? Theme.accent : Theme.border, lineWidth: 1))
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 16)
        }
    }

    // MARK: On-view management

    /// Keep the on-view set consistent with the current selection (order + cap),
    /// preserving still-shown cameras and appending newly selected ones until the
    /// cap. Mirrors Timeline.tsx's onView effect.
    private func syncOnView() {
        let sel = selectedCameras
        let selSet = Set(sel)
        var kept = onView.filter { selSet.contains($0) }
        for name in sel {
            if kept.count >= Self.maxSyncPlayers { break }
            if !kept.contains(name) { kept.append(name) }
        }
        let ordered = sel.filter { kept.contains($0) }
        let capped = Array(ordered.prefix(Self.maxSyncPlayers))
        if capped != onView { onView = capped }
    }

    private func toggleView(_ camera: String) {
        if onView.contains(camera) {
            onView.removeAll { $0 == camera }
        } else {
            addToView(camera)
        }
    }

    private func addToView(_ camera: String) {
        guard selectedCameras.contains(camera), !onView.contains(camera) else { return }
        var next = onView + [camera]
        if next.count > Self.maxSyncPlayers { next = Array(next.suffix(Self.maxSyncPlayers)) }
        onView = selectedCameras.filter { next.contains($0) }
    }

    private func removeFromView(_ camera: String) {
        onView.removeAll { $0 == camera }
    }

    /// Rebuild the export-cams set + reconcile the coordinator's tiles when the
    /// on-view set changes (diff — kept tiles keep playing).
    private func reconcileOnViewTiles() {
        exportCams = Set(onView)
        guard let api = session.api else { return }
        coordinator.setOnView(api: api, cameras: gridCameras(for: onView), dayStart: dayStart)
    }

    private func gridCameras(for names: [String]) -> [SyncGridCamera] {
        names.map { name in
            SyncGridCamera(
                camera: name,
                friendlyName: friendlyName(for: name),
                segments: dayData[name]?.index?.segments ?? []
            )
        }
    }

    // MARK: Range export

    @ViewBuilder
    private var rangeExportBar: some View {
        if let range = exportRange, range.upperBound - range.lowerBound >= 1 {
            let span = range.upperBound - range.lowerBound
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Export \(TimelineTime.spanLabel(span))")
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(Theme.textPrimary)
                        Text("\(TimelineTime.clockLabel(range.lowerBound)) – \(TimelineTime.clockLabel(range.upperBound))\(span >= TimelineTime.maxExportSeconds ? " · max 30 min" : "")")
                            .font(.caption)
                            .foregroundStyle(Theme.textSecondary)
                    }
                    Spacer()
                    Button("Clear") { exportRange = nil }
                        .font(.footnote)
                        .buttonStyle(.bordered)
                        .tint(Theme.accent)
                }
                exportCamPicker(range: range)
            }
            .padding(12)
            .background(Theme.cardBackground())
            .padding(.horizontal, 16)
        } else {
            Text("Drag across the timeline to select a span to export (up to 30 minutes).")
                .font(.caption)
                .foregroundStyle(Theme.textSecondary)
                .padding(.horizontal, 16)
        }
    }

    /// One ShareLink per chosen on-view camera (iOS needs a concrete item per
    /// share, so exports are offered individually). Mirrors the web exportCams.
    @ViewBuilder
    private func exportCamPicker(range: ClosedRange<Double>) -> some View {
        if let api = session.api, !onView.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                Text("Export cameras")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(Theme.textSecondary)
                ForEach(onView, id: \.self) { name in
                    let chosen = exportCams.contains(name)
                    HStack(spacing: 10) {
                        Button {
                            if chosen { exportCams.remove(name) } else { exportCams.insert(name) }
                        } label: {
                            Image(systemName: chosen ? "checkmark.circle.fill" : "circle")
                                .foregroundStyle(chosen ? Theme.accent : Theme.textSecondary)
                        }
                        .buttonStyle(.plain)
                        Text(friendlyName(for: name))
                            .font(.footnote)
                            .foregroundStyle(Theme.textPrimary)
                        Spacer()
                        if chosen {
                            ShareLink(
                                item: api.recordingExportURL(
                                    camera: name,
                                    start: range.lowerBound.rounded(.down),
                                    end: range.upperBound.rounded(.down)
                                )
                            ) {
                                Label("Share", systemImage: "square.and.arrow.up")
                                    .font(.footnote.weight(.semibold))
                            }
                            .tint(Theme.accent)
                        }
                    }
                }
            }
        }
    }

    // MARK: Data loading

    private func loadCatalog() async {
        guard recCameras == nil, let api = session.api else { return }
        do {
            async let camsTask = api.recordingCameras()
            async let groupsTask = api.groups()
            let cams = try await camsTask
            groups = (try? await groupsTask) ?? []

            // Resolve the selection BEFORE anchoring the date. anchorDateToSelection
            // derives the day from the newest footage across the SELECTED set, so
            // resolving afterwards would open the timeline on the newest day across
            // all 12 cameras — quite possibly a day the one selected camera has no
            // coverage for, which looks identical to a broken timeline.
            let stored = TimelineSelectionStore.load()
            var resolved = Self.resolveSelection(stored, for: cams)
            // A stored .group cannot be validated by resolveSelection — group
            // membership lives in `groups`, which is fetched here and whose
            // failure is swallowed above. If the group was deleted (or simply
            // did not load), the selection resolves to NO cameras: no tiles, a
            // dead transport, and a selector that hides the Groups section, so
            // the user cannot even see what went wrong. Fall back to the
            // default rather than showing an unexplainable empty timeline.
            if Self.cameras(for: resolved, cams: cams, groups: groups).isEmpty {
                resolved = Self.resolveSelection(nil, for: cams)
            }
            let selectionChanged = resolved != selection
            // Only a value the USER never picked counts as the default.
            applyingDefault = selectionChanged && resolved != stored
            selection = resolved

            recCameras = cams
            errorMessage = nil
            // When the selection changed, .onChange(of: selection) runs exactly
            // this trio; running it here as well would duplicate it outright.
            //
            // Note this does NOT make cold launch a single fetch: whichever
            // path runs, anchorDateToSelection assigns `date`, and
            // .onChange(of: date) issues its own loadDay. That extra fetch
            // predates this change; requestSeq discards the loser so it is
            // wasted load, not incorrect state. Collapsing it means reworking
            // the date-anchoring flow, which is not in scope here.
            if !selectionChanged {
                syncOnView()
                anchorDateToSelection()
                await loadDay()
            }
        } catch {
            session.handleAPIError(error)
            errorMessage = (error as? ApiError)?.message ?? error.localizedDescription
        }
    }

    /// Anchor the day to the most recent footage of the selected set.
    private func anchorDateToSelection() {
        guard let recCameras else { return }
        let selected = Set(selectedCameras)
        let latest = recCameras
            .filter { selected.contains($0.camera) }
            .compactMap(\.latest)
            .max()
        if let latest {
            date = Date(timeIntervalSince1970: latest)
        } else if !dateAnchored {
            date = Date()
        }
        dateAnchored = true
    }

    /// Fetch each selected camera's recording index + events for the day (in
    /// parallel; a single camera failing just gets an empty lane), then hand the
    /// on-view cameras' segments to the coordinator and issue one aligning seek.
    private func loadDay() async {
        guard let api = session.api else { return }
        let cameras = selectedCameras
        guard !cameras.isEmpty else {
            dayData = [:]
            onView = []
            coordinator.clear()
            return
        }
        requestSeq += 1
        let seq = requestSeq
        loading = true
        let dateStr = TimelineTime.dateString(for: date)
        let ds = dayStart
        let de = dayEnd

        let results = await withTaskGroup(
            of: (String, CameraDayData).self,
            returning: [String: CameraDayData].self
        ) { group in
            for name in cameras {
                group.addTask {
                    // Coverage + marks load independently: the marks are
                    // cosmetic, so a failing events call must not cost us the
                    // footage lane.
                    async let indexTask = api.recordingIndex(camera: name, date: dateStr)
                    async let eventsTask = api.events(
                        camera: name, after: ds, before: de, limit: 1000
                    )
                    let index = try? await indexTask
                    let eventTimes = ((try? await eventsTask)?.events ?? []).map(\.startTime)
                    return (name, CameraDayData(index: index, eventTimes: eventTimes))
                }
            }
            var map: [String: CameraDayData] = [:]
            for await (name, data) in group { map[name] = data }
            return map
        }

        guard seq == requestSeq else { return }
        dayData = results
        loading = false
        syncOnView()

        // Default the playhead to the end of the day's coverage.
        let coverageEnd = cameras
            .compactMap { results[$0]?.index?.ranges.last?.end }
            .max()
        let playhead = coverageEnd.map { TimelineTime.clamp($0 - 2, ds, de - 1) } ?? ds

        coordinator.configureDay(
            api: api,
            cameras: gridCameras(for: onView),
            dayStart: ds,
            newestCoverageEnd: coverageEnd,
            playhead: playhead
        )
        exportCams = Set(onView)
    }
}

// MARK: - Supporting types

/// Which cameras the timeline aggregates (mirrors the web's selector).
enum TimelineSelection: Hashable {
    case all
    case group(Int)
    case custom(Set<String>)
}

/// Persistence for the timeline's camera selection, mirroring the web client's
/// own `vigilume.timeline.cameras` localStorage entry. Related, NOT shared: the
/// web key stores a plain camera-name array, this one stores an all/group/custom
/// selection, so the two are migrated and parsed independently.
///
/// THE POINT OF THE OPTIONAL: `.all` is BOTH a legitimate user choice and what
/// the view falls back to before the camera list arrives. A non-optional
/// persisted value could not tell "the user deliberately picked All" from
/// "the user has never picked anything" — and every user who chose All would
/// be silently re-defaulted to one camera on the next launch. `nil` means
/// never chosen; that is the only state the first-camera default applies to.
enum TimelineSelectionStore {
    private static let key = "vigilume.timeline.selection"
    /// Pre-rename key (the app shipped as "Sentinel"). Migrated on read so a
    /// user's camera pick survives the rename instead of silently resetting to
    /// the first-camera default.
    private static let legacyKey = "sentinel.timeline.selection"

    static func load() -> TimelineSelection? {
        guard let raw = storedRaw() else { return nil }
        if raw == "all" { return .all }
        if raw.hasPrefix("group:"), let id = Int(raw.dropFirst(6)) { return .group(id) }
        if raw.hasPrefix("custom:") {
            let names = raw.dropFirst(7)
                .split(separator: "\u{1F}", omittingEmptySubsequences: true)
                .map(String.init)
            return names.isEmpty ? nil : .custom(Set(names))
        }
        return nil          // unrecognised/corrupt -> treat as never chosen
    }

    /// The stored raw string, adopting the pre-rename key when the new one is
    /// absent. Idempotent — the legacy key is dropped once copied — and nil
    /// when neither key exists, which is exactly "never chosen".
    private static func storedRaw() -> String? {
        let defaults = UserDefaults.standard
        if let raw = defaults.string(forKey: key) { return raw }
        guard let legacy = defaults.string(forKey: legacyKey) else { return nil }
        defaults.set(legacy, forKey: key)
        defaults.removeObject(forKey: legacyKey)
        return legacy
    }

    static func save(_ selection: TimelineSelection) {
        let raw: String
        switch selection {
        case .all:
            raw = "all"
        case .group(let id):
            raw = "group:\(id)"
        case .custom(let names):
            // UNIT SEPARATOR, not a comma: camera names are free text and a
            // comma in one would split it into two phantom cameras on load.
            raw = "custom:" + names.sorted().joined(separator: "\u{1F}")
        }
        UserDefaults.standard.set(raw, forKey: key)
    }
}

enum TimelineZoom {
    case day
    case hour
}

struct CameraDayData {
    let index: RecordingIndex?
    /// Event start times (epoch s) for the day — inert location marks on the
    /// bar, nothing more (tapping an event lives on the Events tab).
    let eventTimes: [Double]
}

// MARK: - Custom camera picker sheet

private struct CustomCameraPickerSheet: View {
    @Environment(\.dismiss) private var dismiss
    let cameras: [RecordingCamera]
    @Binding var selection: TimelineSelection

    private var chosen: Set<String> {
        if case .custom(let names) = selection { return names }
        return Set(cameras.map(\.camera))
    }

    var body: some View {
        NavigationStack {
            List(cameras) { cam in
                let isOn = chosen.contains(cam.camera)
                Button {
                    var next = chosen
                    if isOn { next.remove(cam.camera) } else { next.insert(cam.camera) }
                    selection = .custom(next)
                } label: {
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(cam.friendlyName.isEmpty ? cam.camera : cam.friendlyName)
                                .foregroundStyle(Theme.textPrimary)
                            if !cam.hasRecordings {
                                Text("No footage")
                                    .font(.caption)
                                    .foregroundStyle(Theme.textSecondary)
                            }
                        }
                        Spacer()
                        Image(systemName: isOn ? "checkmark.circle.fill" : "circle")
                            .foregroundStyle(isOn ? Theme.accent : Theme.textSecondary)
                    }
                }
                .listRowBackground(Theme.surface)
            }
            .scrollContentBackground(.hidden)
            .background(Theme.bg)
            .navigationTitle("Choose cameras")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                        .tint(Theme.accent)
                }
            }
        }
    }
}
