import SwiftUI

/// Events tab: paged event list with Camera/Object dropdown filters, snapshot
/// thumbnails, pull-to-refresh, and live prepend from the /api/ws socket.
/// Deep links (vigilume://events/<id>, push taps) land here via
/// session.pendingEventID and push EventDetailView.
struct EventsView: View {
    @EnvironmentObject private var session: SessionModel

    @State private var events: [Event] = []
    @State private var total = 0
    @State private var cameras: [Camera] = []
    @State private var selectedCamera: String?
    @State private var selectedLabel: String?
    /// nil == no date filter. When set, events are constrained to this day
    /// between `fromTime` and `toTime` (mapped to after/before epoch seconds).
    @State private var filterDate: Date?
    @State private var fromTime = Calendar.current.startOfDay(for: Date())
    @State private var toTime = Calendar.current.date(
        bySettingHour: 23, minute: 59, second: 59, of: Date()
    ) ?? Date()
    @State private var showingDateFilter = false
    @State private var initialLoading = true
    @State private var loadingMore = false
    @State private var errorMessage: String?
    @State private var requestSeq = 0
    @State private var path = NavigationPath()

    private static let pageSize = 50
    private static let defaultLabels = ["person", "dog", "cat", "car"]

    var body: some View {
        NavigationStack(path: $path) {
            VStack(spacing: 0) {
                filterBar
                content
            }
            .background(Theme.bg)
            .navigationTitle("Events")
            .navigationDestination(for: Int.self) { id in
                EventDetailView(eventID: id)
            }
            .task { await initialLoad() }
            .refreshable { await reload() }
            .onChange(of: selectedCamera) { _, _ in Task { await reload() } }
            .onChange(of: selectedLabel) { _, _ in Task { await reload() } }
            .onChange(of: filterDate) { _, _ in Task { await reload() } }
            .onChange(of: fromTime) { _, _ in if filterDate != nil { Task { await reload() } } }
            .onChange(of: toTime) { _, _ in if filterDate != nil { Task { await reload() } } }
            .onReceive(session.wsMessages) { handleSocket($0) }
            .onReceive(session.wsReconnected) { _ in Task { await reload() } }
            .onReceive(NotificationCenter.default.publisher(for: .vigilumeEventDeleted)) { note in
                guard let id = note.object as? Int else { return }
                if let index = events.firstIndex(where: { $0.id == id }) {
                    events.remove(at: index)
                    total = max(0, total - 1)
                }
            }
            .onChange(of: session.pendingEventID) { _, newValue in
                consumePendingDeepLink(newValue)
            }
            .onAppear {
                consumePendingDeepLink(session.pendingEventID)
            }
        }
    }

    // MARK: Filter dropdowns (Camera / Object)

    private var knownLabels: [String] {
        var set = Set(Self.defaultLabels)
        for event in events { set.formUnion(event.allLabels) }
        if let selectedLabel { set.insert(selectedLabel) }
        return set.sorted()
    }

    /// Compact filter row: Camera / Object dropdowns plus a Date button that
    /// toggles an inline day + time-window panel below.
    private var filterBar: some View {
        VStack(spacing: 0) {
            HStack(spacing: 10) {
                filterMenu(
                    icon: "video",
                    title: selectedCamera.map(friendlyName(for:)) ?? "All cameras",
                    isActive: selectedCamera != nil,
                    accessibility: "Camera filter"
                ) {
                    Picker("Camera", selection: $selectedCamera) {
                        Text("All cameras").tag(String?.none)
                        ForEach(cameras) { camera in
                            Text(camera.friendlyName.isEmpty ? camera.name : camera.friendlyName)
                                .tag(String?.some(camera.name))
                        }
                    }
                }
                filterMenu(
                    icon: "tag",
                    title: selectedLabel?.capitalized ?? "All objects",
                    isActive: selectedLabel != nil,
                    accessibility: "Object filter"
                ) {
                    Picker("Object", selection: $selectedLabel) {
                        Text("All objects").tag(String?.none)
                        ForEach(knownLabels, id: \.self) { label in
                            Text(label.capitalized).tag(String?.some(label))
                        }
                    }
                }
                dateFilterButton
                Spacer(minLength: 0)
            }
            if showingDateFilter {
                dateFilterPanel
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(Theme.bgDeep)
        .overlay(alignment: .bottom) {
            Rectangle().fill(Theme.border).frame(height: 1)
        }
    }

    /// Capsule button mirroring `filterMenu`'s look; taps expand the date panel.
    private var dateFilterButton: some View {
        Button {
            showingDateFilter.toggle()
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "calendar")
                    .font(.caption)
                Text(filterDate.map { $0.formatted(date: .abbreviated, time: .omitted) } ?? "Any date")
                    .font(.subheadline.weight(.medium))
                    .lineLimit(1)
                Image(systemName: "chevron.down")
                    .font(.caption2)
            }
            .foregroundStyle(filterDate != nil ? Theme.bgDeep : Theme.textPrimary)
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(Capsule().fill(filterDate != nil ? Theme.accent : Theme.surface))
            .overlay(Capsule().stroke(filterDate != nil ? Theme.accent : Theme.border, lineWidth: 1))
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Date filter")
    }

    /// Inline day + from/to time-window pickers with a Clear/Done footer.
    private var dateFilterPanel: some View {
        VStack(spacing: 10) {
            DatePicker(
                "Date",
                selection: Binding(
                    get: { filterDate ?? Calendar.current.startOfDay(for: Date()) },
                    set: { filterDate = $0 }
                ),
                displayedComponents: .date
            )
            DatePicker("From", selection: $fromTime, displayedComponents: .hourAndMinute)
            DatePicker("To", selection: $toTime, displayedComponents: .hourAndMinute)
            HStack {
                Button("Clear") {
                    filterDate = nil
                    showingDateFilter = false
                }
                .foregroundStyle(filterDate != nil ? Theme.danger : Theme.textSecondary)
                .disabled(filterDate == nil)
                Spacer()
                Button("Done") { showingDateFilter = false }
                    .foregroundStyle(Theme.accent)
            }
            .font(.subheadline.weight(.medium))
        }
        .tint(Theme.accent)
        .font(.subheadline)
        .foregroundStyle(Theme.textPrimary)
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(Theme.surface))
        .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).stroke(Theme.border, lineWidth: 1))
        .padding(.top, 10)
    }

    /// Maps the chosen day + from/to time to an after/before epoch-seconds
    /// window; nil when no date is selected (no date constraint applied).
    private var dateWindow: (after: Double, before: Double)? {
        guard let filterDate else { return nil }
        let cal = Calendar.current
        let day = cal.startOfDay(for: filterDate)
        let from = cal.dateComponents([.hour, .minute], from: fromTime)
        let to = cal.dateComponents([.hour, .minute], from: toTime)
        guard
            let after = cal.date(
                bySettingHour: from.hour ?? 0, minute: from.minute ?? 0, second: 0, of: day
            ),
            let before = cal.date(
                bySettingHour: to.hour ?? 23, minute: to.minute ?? 59, second: 59, of: day
            )
        else { return nil }
        return (after.timeIntervalSince1970, before.timeIntervalSince1970)
    }

    private func filterMenu<Content: View>(
        icon: String,
        title: String,
        isActive: Bool,
        accessibility: String,
        @ViewBuilder content: () -> Content
    ) -> some View {
        Menu {
            content()
        } label: {
            HStack(spacing: 6) {
                Image(systemName: icon)
                    .font(.caption)
                Text(title)
                    .font(.subheadline.weight(.medium))
                    .lineLimit(1)
                Image(systemName: "chevron.down")
                    .font(.caption2)
            }
            .foregroundStyle(isActive ? Theme.bgDeep : Theme.textPrimary)
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(Capsule().fill(isActive ? Theme.accent : Theme.surface))
            .overlay(Capsule().stroke(isActive ? Theme.accent : Theme.border, lineWidth: 1))
        }
        .accessibilityLabel(accessibility)
    }

    // MARK: List

    @ViewBuilder
    private var content: some View {
        if let errorMessage, events.isEmpty {
            ContentUnavailableView(
                "Couldn't load events",
                systemImage: "bell.slash",
                description: Text(errorMessage)
            )
            .frame(maxHeight: .infinity)
        } else if events.isEmpty && !initialLoading {
            ContentUnavailableView(
                "No events",
                systemImage: "bell.badge",
                description: Text(hasFilters ? "Nothing matches these filters yet" : "Detections will appear here")
            )
            .frame(maxHeight: .infinity)
        } else {
            List {
                ForEach(events) { event in
                    NavigationLink(value: event.id) {
                        EventRowView(event: event, cameraName: friendlyName(for: event.camera))
                    }
                    .listRowBackground(Theme.surface)
                    .onAppear {
                        if event.id == events.last?.id {
                            Task { await loadMore() }
                        }
                    }
                }
                if loadingMore {
                    HStack {
                        Spacer()
                        ProgressView().tint(Theme.accent)
                        Spacer()
                    }
                    .listRowBackground(Color.clear)
                }
            }
            .listStyle(.plain)
            .scrollContentBackground(.hidden)
            .overlay {
                if initialLoading && events.isEmpty {
                    ProgressView().tint(Theme.accent)
                }
            }
        }
    }

    private var hasFilters: Bool { selectedCamera != nil || selectedLabel != nil || filterDate != nil }

    private func friendlyName(for camera: String) -> String {
        cameras.first(where: { $0.name == camera })
            .map { $0.friendlyName.isEmpty ? camera : $0.friendlyName } ?? camera
    }

    private func consumePendingDeepLink(_ id: Int?) {
        guard let id else { return }
        session.pendingEventID = nil
        path.append(id)
    }

    // MARK: Loading

    private func initialLoad() async {
        if cameras.isEmpty, let api = session.api {
            cameras = (try? await api.cameras()) ?? []
        }
        if events.isEmpty { await reload() }
    }

    private func reload() async {
        await load(reset: true)
    }

    private func loadMore() async {
        guard !loadingMore, !initialLoading, events.count < total else { return }
        await load(reset: false)
    }

    private func load(reset: Bool) async {
        guard let api = session.api else { return }
        requestSeq += 1
        let seq = requestSeq
        if reset { initialLoading = events.isEmpty } else { loadingMore = true }
        do {
            let window = dateWindow
            let page = try await api.events(
                camera: selectedCamera,
                label: selectedLabel,
                after: window?.after,
                before: window?.before,
                limit: Self.pageSize,
                offset: reset ? 0 : events.count
            )
            guard seq == requestSeq else { return }
            total = page.total
            if reset {
                events = page.events
            } else {
                let known = Set(events.map(\.id))
                events.append(contentsOf: page.events.filter { !known.contains($0.id) })
            }
            errorMessage = nil
        } catch {
            guard seq == requestSeq else { return }
            session.handleAPIError(error)
            errorMessage = (error as? ApiError)?.message ?? error.localizedDescription
        }
        if seq == requestSeq {
            initialLoading = false
            loadingMore = false
        }
    }

    // MARK: Live socket updates

    private func handleSocket(_ message: WSMessage) {
        switch message {
        case .eventNew(let event), .doorbell(let event):
            guard matchesFilters(event) else { return }
            if let index = events.firstIndex(where: { $0.id == event.id }) {
                events[index] = event
            } else {
                events.insert(event, at: 0)
                total += 1
            }
        case .eventUpdate(let event), .eventEnd(let event):
            if let index = events.firstIndex(where: { $0.id == event.id }) {
                events[index] = event
            }
        default:
            break
        }
    }

    private func matchesFilters(_ event: Event) -> Bool {
        if let selectedCamera, event.camera != selectedCamera { return false }
        if let selectedLabel, !event.allLabels.contains(selectedLabel) { return false }
        if let window = dateWindow {
            if event.startTime < window.after || event.startTime > window.before { return false }
        }
        return true
    }
}

// MARK: - Row

private struct EventRowView: View {
    @EnvironmentObject private var session: SessionModel
    let event: Event
    let cameraName: String

    var body: some View {
        HStack(spacing: 12) {
            thumbnail
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 8) {
                    // All detected classes (multi-object): a colored dot + name
                    // per label, so "person + car" shows both, not just one.
                    ForEach(event.allLabels, id: \.self) { name in
                        HStack(spacing: 4) {
                            Circle()
                                .fill(EventLabelStyle.color(for: name))
                                .frame(width: 8, height: 8)
                            Text(name.capitalized)
                                .font(.body.weight(.medium))
                                .foregroundStyle(Theme.textPrimary)
                                .lineLimit(1)
                        }
                    }
                    if event.count > 1 {
                        Text("×\(event.count)")
                            .font(.caption.bold())
                            .foregroundStyle(Theme.accent)
                    }
                    if event.endTime == nil {
                        Text("LIVE")
                            .font(.caption2.bold())
                            .foregroundStyle(Theme.danger)
                    }
                }
                Text(cameraName)
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
                Text(Date(timeIntervalSince1970: event.startTime)
                        .formatted(date: .abbreviated, time: .shortened))
                    .font(.caption2)
                    .foregroundStyle(Theme.textSecondary)
            }
            Spacer(minLength: 0)
        }
        .padding(.vertical, 4)
    }

    private var thumbnail: some View {
        Group {
            if event.hasSnapshot, let url = session.api?.eventSnapshotURL(id: event.id) {
                AsyncImage(url: url) { image in
                    image.resizable().aspectRatio(contentMode: .fill)
                } placeholder: {
                    thumbPlaceholder
                }
            } else {
                thumbPlaceholder
            }
        }
        .frame(width: 106, height: 60)
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(Theme.border, lineWidth: 1)
        )
    }

    private var thumbPlaceholder: some View {
        Rectangle()
            .fill(Theme.surfaceAlt)
            .overlay(
                Image(systemName: "photo")
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
            )
    }
}

extension Notification.Name {
    /// Posted (object = event id as Int) after an event is deleted so the
    /// list can drop the row without a refetch.
    static let vigilumeEventDeleted = Notification.Name("vigilume.eventDeleted")
}
