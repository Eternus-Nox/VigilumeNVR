import SwiftUI
import AVKit

/// Event detail — the push-notification / deep-link landing target.
/// The media area is driven by the backend's `clip_state` (docs/CONTRACTS.md)
/// so we never show a silently broken player: a ready clip AUTOPLAYS in
/// AVPlayer (opening the event is the user's gesture); a "processing" clip
/// shows the camera's LIVE stream with a "Clip processing…" badge while we
/// refetch with backoff — when the clip lands we don't yank the live view,
/// we offer "Clip ready — tap to watch" which swaps to the autoplaying clip;
/// disabled/unavailable states say plainly why there is no video.
struct EventDetailView: View {
    @EnvironmentObject private var session: SessionModel
    @Environment(\.dismiss) private var dismiss

    let eventID: Int

    @State private var detail: EventDetail?
    @State private var errorMessage: String?
    @State private var player: AVPlayer?
    /// The clip became ready while the user was watching live — hold the live
    /// view and show the "tap to watch" affordance instead of hard-swapping.
    @State private var clipAwaitingTap = false
    @State private var pollStalled = false
    @State private var pollGeneration = 0
    @State private var confirmDelete = false
    @State private var deleting = false
    @State private var deleteError: String?
    @State private var confirmReject = false
    @State private var rejecting = false
    @State private var rejectError: String?
    /// Presents the event clip full-screen (landscape, tap to dismiss).
    @State private var showClipFullScreen = false

    // The only explicit save flow: the event clip → photo library.
    @StateObject private var clipSaver = MediaSaver()

    /// Backoff schedule (seconds) for polling a clip still being cut
    /// (~20 s after event end, per the recorder). Caps out so we stop.
    private static let pollDelays: [Double] = [2.5, 3.5, 5, 7, 10, 15]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if let detail {
                    media(for: detail)
                    clipStatus(for: detail)
                    saveCard(for: detail)
                    metadata(for: detail)
                    rejectCard(for: detail)
                } else if let errorMessage {
                    ContentUnavailableView(
                        "Event unavailable",
                        systemImage: "bell.slash",
                        description: Text(errorMessage)
                    )
                    .padding(.top, 40)
                } else {
                    ProgressView().tint(Theme.accent)
                        .frame(maxWidth: .infinity)
                        .padding(.top, 60)
                }
            }
            .padding()
        }
        .background(Theme.bg)
        .navigationTitle(titleText)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar { toolbarContent }
        .task { await loadDetail() }
        .onDisappear { player?.pause() }
        .fullScreenCover(isPresented: $showClipFullScreen) {
            if let api = session.api, let detail, clipReady(detail) {
                EventVideoFullScreenView(url: api.eventClipURL(id: detail.id))
            }
        }
        .alert("Delete event", isPresented: $confirmDelete) {
            Button("Delete", role: .destructive) { Task { await deleteEvent() } }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Delete this event and its media? This cannot be undone.")
        }
        .alert("Delete failed", isPresented: .init(
            get: { deleteError != nil },
            set: { if !$0 { deleteError = nil } }
        )) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(deleteError ?? "")
        }
        .alert(rejectPrompt, isPresented: $confirmReject) {
            Button("Exclude", role: .destructive) { Task { await rejectEvent() } }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This stops alerting on detections like it here, and removes this event.")
        }
        .alert("Couldn't exclude", isPresented: .init(
            get: { rejectError != nil },
            set: { if !$0 { rejectError = nil } }
        )) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(rejectError ?? "")
        }
    }

    /// Confirm-alert title, e.g. "Not a real Person?" — mirrors the reject
    /// button's label with the primary detected class.
    private var rejectPrompt: String {
        "Not a real \(detail?.label.capitalized ?? "detection")?"
    }

    private var titleText: String {
        guard let detail else { return "Event #\(eventID)" }
        let count = detail.count > 1 ? " ×\(detail.count)" : ""
        return "\(detail.label.capitalized)\(count)"
    }

    // MARK: Toolbar (delete only — saving lives in the single Save card)

    @ToolbarContentBuilder
    private var toolbarContent: some ToolbarContent {
        ToolbarItemGroup(placement: .topBarTrailing) {
            if detail != nil, session.isAdmin {
                Button(role: .destructive) {
                    confirmDelete = true
                } label: {
                    if deleting {
                        ProgressView()
                    } else {
                        Image(systemName: "trash")
                    }
                }
                .disabled(deleting)
            }
        }
    }

    // MARK: Media

    private func clipReady(_ detail: EventDetail) -> Bool {
        detail.clipState == .ready && detail.hasClip
    }

    @ViewBuilder
    private func media(for detail: EventDetail) -> some View {
        if clipReady(detail), !clipAwaitingTap, let player {
            VideoPlayer(player: player)
                .aspectRatio(16 / 9, contentMode: .fit)
                .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                .overlay(alignment: .topTrailing) {
                    Button {
                        // Hand playback to the full-screen cover — pause the
                        // inline copy so audio doesn't double up behind it.
                        player.pause()
                        showClipFullScreen = true
                    } label: {
                        Image(systemName: "arrow.up.left.and.arrow.down.right")
                            .font(.footnote.weight(.semibold))
                            .foregroundStyle(.white)
                            .frame(width: 32, height: 32)
                            .background(Circle().fill(Color.black.opacity(0.55)))
                    }
                    .buttonStyle(.plain)
                    .padding(8)
                    .accessibilityLabel("Full screen")
                }
        } else if detail.clipState == .processing || clipAwaitingTap {
            liveWhileProcessing(for: detail)
        } else if detail.hasSnapshot {
            snapshotImage(id: detail.id)
        } else {
            Rectangle()
                .fill(Theme.surfaceAlt)
                .aspectRatio(16 / 9, contentMode: .fit)
                .overlay(
                    Text("No media captured for this event")
                        .font(.caption)
                        .foregroundStyle(Theme.textSecondary)
                )
                .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        }

        // Ready clip + snapshot: also show the annotated frame below.
        if clipReady(detail), !clipAwaitingTap, detail.hasSnapshot {
            DisclosureGroup {
                snapshotImage(id: detail.id)
            } label: {
                Text("Annotated snapshot")
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(Theme.textPrimary)
            }
            .tint(Theme.accent)
        }
    }

    /// Live camera stream shown while the clip is still being cut. A small
    /// "Clip processing…" badge sits on the video; once the clip is ready
    /// (clipAwaitingTap) the badge becomes "Clip ready — tap to watch".
    private func liveWhileProcessing(for detail: EventDetail) -> some View {
        EventLiveView(
            streamURL: session.api?.liveStreamURL(camera: detail.camera),
            fallbackURL: session.api?.liveSubStreamURL(camera: detail.camera)
        )
            .overlay(alignment: .topLeading) {
                if !clipAwaitingTap {
                    HStack(spacing: 5) {
                        ProgressView()
                            .controlSize(.mini)
                            .tint(.white)
                        Text("Clip processing…")
                            .font(.caption2.weight(.semibold))
                            .foregroundStyle(.white)
                    }
                    .padding(.horizontal, 9)
                    .padding(.vertical, 5)
                    .background(Capsule().fill(Color.black.opacity(0.55)))
                    .padding(8)
                }
            }
            .overlay(alignment: .bottom) {
                if clipAwaitingTap {
                    Button {
                        clipAwaitingTap = false
                        player?.play()
                    } label: {
                        Label("Clip ready — tap to watch", systemImage: "play.fill")
                            .font(.footnote.weight(.semibold))
                            .foregroundStyle(Theme.bgDeep)
                            .padding(.horizontal, 14)
                            .padding(.vertical, 8)
                            .background(Capsule().fill(Theme.accent))
                    }
                    .buttonStyle(.plain)
                    .padding(.bottom, 10)
                }
            }
    }

    private func snapshotImage(id: Int) -> some View {
        AsyncImage(url: session.api?.eventSnapshotURL(id: id)) { image in
            image.resizable().aspectRatio(contentMode: .fit)
        } placeholder: {
            Rectangle().fill(Theme.surfaceAlt)
                .aspectRatio(16 / 9, contentMode: .fit)
                .overlay(ProgressView().tint(Theme.accent))
        }
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    // MARK: Clip lifecycle status

    @ViewBuilder
    private func clipStatus(for detail: EventDetail) -> some View {
        switch detail.clipState {
        case .ready:
            EmptyView()
        case .processing:
            HStack(spacing: 12) {
                ProgressView().tint(Theme.accent)
                VStack(alignment: .leading, spacing: 2) {
                    Text(pollStalled ? "Still processing recording…" : "Processing recording…")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(Theme.textPrimary)
                    Text(pollStalled
                        ? "The clip is taking longer than usual to cut from continuous recording."
                        : "The clip is being cut from continuous recording — this usually takes about half a minute.")
                        .font(.caption)
                        .foregroundStyle(Theme.textSecondary)
                }
                Spacer()
                Button("Refresh") {
                    pollStalled = false
                    Task { await refetch(restartPolling: true) }
                }
                .font(.caption.weight(.semibold))
                .buttonStyle(.bordered)
                .tint(Theme.accent)
            }
            .padding(12)
            .background(Theme.cardBackground())
        case .recordingDisabled:
            statusCard(detail.hasSnapshot
                ? "Recording is off for this camera, so no clip was saved — the annotated snapshot is shown above."
                : "Recording is off for this camera, so no clip was saved.")
        case .unavailable:
            statusCard("No recording was saved for this event.")
        }
    }

    // MARK: Save (single action — the event CLIP to Photos, nothing else)

    /// One clear action: download the ready clip (mp4) and add it to the photo
    /// library. Snapshot saving was removed deliberately — clip → Photos only.
    @ViewBuilder
    private func saveCard(for detail: EventDetail) -> some View {
        if let api = session.api, clipReady(detail) {
            VStack(alignment: .leading, spacing: 12) {
                Label("Save", systemImage: "square.and.arrow.down")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(Theme.textPrimary)

                PhotosSaveButton(
                    idleTitle: "Save clip to Photos",
                    savedTitle: "Clip saved to Photos",
                    systemImage: "film",
                    saver: clipSaver
                ) {
                    clipSaver.save(from: api.eventClipURL(id: detail.id), kind: .video)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(14)
            .background(Theme.cardBackground())
        }
    }

    private func statusCard(_ text: String) -> some View {
        Text(text)
            .font(.caption)
            .foregroundStyle(Theme.textSecondary)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(12)
            .background(Theme.cardBackground())
    }

    // MARK: Metadata

    private func metadata(for detail: EventDetail) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            metaRow("Camera", detail.camera.replacingOccurrences(of: "_", with: " ").capitalized)
            // All detected classes (multi-object), not just the primary label.
            metaRow(
                detail.allLabels.count > 1 ? "Objects" : "Label",
                detail.allLabels.map(\.capitalized).joined(separator: ", ")
            )
            if detail.count > 0 {
                metaRow("Count", "\(detail.count) in frame")
            }
            metaRow("Score", String(format: "%.0f%%", detail.score * 100))
            metaRow("Start", format(epoch: detail.startTime))
            metaRow("End", detail.endTime.map(format(epoch:)) ?? "in progress")
            if let end = detail.endTime {
                metaRow("Duration", formatDuration(end - detail.startTime))
            }
            if !detail.zones.isEmpty {
                metaRow("Zones", detail.zones.map(\.capitalized).joined(separator: ", "))
            }
            metaRow("Event ID", "\(detail.id)")
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(Theme.cardBackground())
    }

    private func metaRow(_ title: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(title)
                .font(.caption)
                .foregroundStyle(Theme.textSecondary)
                .frame(width: 80, alignment: .leading)
            Text(value)
                .font(.subheadline)
                .foregroundStyle(Theme.textPrimary)
        }
    }

    private func format(epoch: Double) -> String {
        Date(timeIntervalSince1970: epoch)
            .formatted(date: .abbreviated, time: .standard)
    }

    private func formatDuration(_ seconds: Double) -> String {
        let s = Int(seconds.rounded())
        if s >= 3600 { return "\(s / 3600)h \((s % 3600) / 60)m" }
        if s >= 60 { return "\(s / 60)m \(s % 60)s" }
        return "\(s)s"
    }

    // MARK: Reject (admin-only "false detection" → learn a suppression)

    /// Admin-only card that marks this event a false detection: rejecting it
    /// learns a suppression (so similar detections stop alerting) and deletes
    /// the event. Undoable under Settings › Excluded objects.
    @ViewBuilder
    private func rejectCard(for detail: EventDetail) -> some View {
        if session.isAdmin {
            Button(role: .destructive) {
                confirmReject = true
            } label: {
                HStack(spacing: 8) {
                    if rejecting {
                        ProgressView().controlSize(.small).tint(Theme.danger)
                    } else {
                        Image(systemName: "hand.thumbsdown")
                    }
                    Text("Not a real \(detail.label.capitalized)")
                    Spacer()
                }
                .font(.callout.weight(.medium))
                .foregroundStyle(Theme.danger)
                .padding(.horizontal, 12)
                .frame(maxWidth: .infinity)
                .frame(height: 40)
                .background(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .fill(Theme.danger.opacity(0.10))
                )
            }
            .buttonStyle(.plain)
            .disabled(rejecting)
        }
    }

    // MARK: Loading + processing poll

    private func loadDetail() async {
        guard let api = session.api else { return }
        do {
            let loaded = try await api.event(id: eventID)
            apply(loaded, api: api)
            if loaded.clipState == .processing {
                await pollWhileProcessing()
            }
        } catch {
            session.handleAPIError(error)
            errorMessage = (error as? ApiError)?.message ?? error.localizedDescription
        }
    }

    /// Refetch once; optionally restart the backoff poll (manual Refresh).
    private func refetch(restartPolling: Bool) async {
        guard let api = session.api else { return }
        guard let loaded = try? await api.event(id: eventID) else { return }
        apply(loaded, api: api)
        if restartPolling, loaded.clipState == .processing {
            await pollWhileProcessing()
        }
    }

    /// Auto-poll with backoff while the clip is being cut, then give up
    /// gracefully (pollStalled shows the "taking longer than usual" copy).
    private func pollWhileProcessing() async {
        pollGeneration += 1
        let generation = pollGeneration
        for delay in Self.pollDelays {
            try? await Task.sleep(for: .seconds(delay))
            guard !Task.isCancelled, generation == pollGeneration else { return }
            guard detail?.clipState == .processing else { return }
            await refetch(restartPolling: false)
            guard generation == pollGeneration else { return }
            if detail?.clipState != .processing { return }
        }
        if generation == pollGeneration { pollStalled = true }
    }

    private func apply(_ loaded: EventDetail, api: APIClient) {
        let wasProcessing = detail?.clipState == .processing
        let wasReady = detail.map(clipReady) ?? false
        detail = loaded
        if clipReady(loaded), !wasReady || player == nil {
            player = AVPlayer(url: api.eventClipURL(id: loaded.id))
            if wasProcessing || clipAwaitingTap {
                // The user is watching live — offer "tap to watch" rather
                // than yanking the live view out from under them.
                clipAwaitingTap = true
            } else {
                // Opening the event was the user's gesture — autoplay.
                player?.play()
            }
        }
    }

    // MARK: Delete

    private func deleteEvent() async {
        guard let api = session.api, let detail else { return }
        deleting = true
        do {
            try await api.deleteEvent(id: detail.id)
            NotificationCenter.default.post(name: .vigilumeEventDeleted, object: detail.id)
            dismiss()
        } catch {
            session.handleAPIError(error)
            deleteError = (error as? ApiError)?.message ?? error.localizedDescription
        }
        deleting = false
    }

    // MARK: Reject

    /// Mark this event a false detection: the backend learns a suppression and
    /// deletes the event. Mirrors `deleteEvent()` — on success drop the row via
    /// .vigilumeEventDeleted and dismiss.
    private func rejectEvent() async {
        guard let api = session.api, let detail else { return }
        rejecting = true
        do {
            try await api.rejectEvent(id: detail.id)
            NotificationCenter.default.post(name: .vigilumeEventDeleted, object: detail.id)
            dismiss()
        } catch {
            session.handleAPIError(error)
            rejectError = (error as? ApiError)?.message ?? error.localizedDescription
        }
        rejecting = false
    }
}

// MARK: - Live view while the clip is processing

/// Minimal live HLS surface for the event-detail media slot: reuses the
/// self-healing LivePlayerModel (same player the Cameras tab uses), muted,
/// attaching on appear and fully detaching on disappear.
private struct EventLiveView: View {
    let streamURL: URL?
    let fallbackURL: URL?

    @StateObject private var model = LivePlayerModel()

    var body: some View {
        ZStack {
            Theme.bgDeep
            if let player = model.player {
                PlayerLayerView(player: player, videoGravity: .resizeAspect)
            }
            if model.state != .playing {
                ProgressView().tint(Theme.accent)
            }
        }
        .aspectRatio(16 / 9, contentMode: .fit)
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .stroke(Theme.border, lineWidth: 1)
        )
        .onAppear { attach() }
        .onDisappear { model.stop() }
        .onChange(of: streamURL) { _, _ in attach() }
    }

    private func attach() {
        guard let streamURL else { return }
        model.play(primary: streamURL, fallback: fallbackURL)
    }
}

// MARK: - Full-screen event clip

/// Full-screen (landscape-friendly) player for a ready event clip, mirroring
/// the live SingleCameraView cover: dark, tap anywhere (or the close button)
/// to dismiss. Unlike live tiles it keeps `.resizeAspect` so the WHOLE recorded
/// scene stays visible (no cropping). Builds its own AVPlayer from the clip URL
/// so it never fights the inline VideoPlayer over one shared player.
private struct EventVideoFullScreenView: View {
    let url: URL

    @Environment(\.dismiss) private var dismiss
    @State private var player: AVPlayer?

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            // Pinch/pan to zoom (1×…4×) the recorded clip; a single tap still
            // dismisses (matches the live cover), double-tap resets zoom.
            ZoomableVideo(onSingleTap: { dismiss() }) {
                if let player {
                    PlayerLayerView(player: player, videoGravity: .resizeAspect)
                } else {
                    Color.clear
                }
            }
            .ignoresSafeArea()

            VStack {
                HStack {
                    Button {
                        dismiss()
                    } label: {
                        Image(systemName: "xmark")
                            .font(.body.weight(.semibold))
                            .foregroundStyle(.white)
                            .frame(width: 36, height: 36)
                            .background(Circle().fill(Color.black.opacity(0.5)))
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("Close")
                    Spacer()
                }
                Spacer()
            }
            .padding(.horizontal, 16)
            .padding(.top, 8)
        }
        .preferredColorScheme(.dark)
        .onAppear {
            LiveAudioSession.activatePlayback()
            let newPlayer = AVPlayer(url: url)
            player = newPlayer
            newPlayer.play()
        }
        .onDisappear {
            player?.pause()
            // Balance the activatePlayback above — RTCAudioSession
            // reference-counts, so leaking this activation would pin the session
            // active and defeat the live view's mic scoping.
            LiveAudioSession.deactivate()
        }
    }
}

// MARK: - PhotosSaveButton

/// One save-to-Photos button driven by a MediaSaver: shows a determinate
/// download bar, then "Saving…", then a green check (or the failure reason
/// below the button, with the button re-armed for retry).
private struct PhotosSaveButton: View {
    let idleTitle: String
    let savedTitle: String
    let systemImage: String
    @ObservedObject var saver: MediaSaver
    let action: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Button(action: action) {
                HStack(spacing: 8) {
                    switch saver.phase {
                    case .idle, .failed:
                        Image(systemName: systemImage)
                        Text(idleTitle)
                    case .requestingAccess:
                        ProgressView().controlSize(.small).tint(Theme.accent)
                        Text("Waiting for Photos permission…")
                    case .downloading(let fraction):
                        ProgressView().controlSize(.small).tint(Theme.accent)
                        Text(fraction.map { "Downloading… \(Int($0 * 100))%" }
                            ?? "Downloading…")
                    case .saving:
                        ProgressView().controlSize(.small).tint(Theme.accent)
                        Text("Saving to Photos…")
                    case .saved:
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(Theme.success)
                        Text(savedTitle)
                            .foregroundStyle(Theme.success)
                    }
                    Spacer()
                }
                .font(.callout.weight(.medium))
                .foregroundStyle(saver.phase == .saved ? Theme.success : Theme.accent)
                .padding(.horizontal, 12)
                .frame(maxWidth: .infinity)
                .frame(height: 40)
                .background(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .fill((saver.phase == .saved ? Theme.success : Theme.accent).opacity(0.10))
                )
            }
            .buttonStyle(.plain)
            .disabled(saver.isBusy)

            if case .downloading(let fraction) = saver.phase, let fraction {
                ProgressView(value: fraction)
                    .tint(Theme.accent)
            }

            if case .failed(let message) = saver.phase {
                Text(message)
                    .font(.caption)
                    .foregroundStyle(Theme.danger)
            }
        }
    }
}
