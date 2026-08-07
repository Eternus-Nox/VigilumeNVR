import AVFoundation
import Combine
import Foundation

/// One on-view camera's inputs for the sync grid (mirrors the web GridCamera).
struct SyncGridCamera {
    let camera: String
    let friendlyName: String
    let segments: [RecordingSegment]
}

/// The MASTER CLOCK + shared transport for the multi-camera Timeline — one
/// instance owned by TimelineView. It is the single source of truth for the
/// shared playhead, play/pause, rate and mute, and it fans those out to one
/// SyncCameraPlayerModel per ON-VIEW camera (each owning its own AVPlayer).
///
/// Because every camera maps the SAME wall-clock target through ITS OWN
/// segments, an explicit seek re-aligns all players precisely (gaps included).
/// During play a single coordinator-chosen LEADER (the first on-view camera with
/// footage at the playhead) advances the shared playhead; a periodic drift check
/// nudges followers back toward it so independent AVPlayers stay locked
/// sub-second without a visible group stutter. Mirrors Timeline.tsx's shared
/// state + leaderCamera + SyncPlaybackGrid.
@MainActor
final class TimelineSyncCoordinator: ObservableObject {
    /// One player per on-view camera, in display order (drives the grid).
    @Published private(set) var models: [SyncCameraPlayerModel] = []
    /// Shared wall-clock playhead (epoch seconds) the bar/readout render.
    @Published var playhead: Double = 0
    @Published private(set) var isPlaying = false
    @Published private(set) var rate: Float = 1
    @Published private(set) var muted = true
    /// True while the user drags the bar — leader follow is suppressed.
    @Published var isScrubbing = false
    /// The camera whose player currently drives the playhead (has footage here).
    @Published private(set) var leaderCamera: String?

    private var api: APIClient?
    private var dayStart: Double = 0
    private var dayEnd: Double = 0
    private var newestCoverageEnd: Double?

    private var driftTimer: Timer?
    private let driftThreshold: Double = 0.75

    init() {
        driftTimer = Timer.scheduledTimer(withTimeInterval: 2, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.driftTick() }
        }
    }

    deinit {
        driftTimer?.invalidate()
    }

    // MARK: Configuration

    /// Full (re)build for a freshly-loaded day: (re)point every on-view camera at
    /// the day's segments and issue ONE aligning seek (paused), exactly like the
    /// web's post-fetch setSeek. Existing player objects are reused by name to
    /// avoid AVPlayer churn.
    func configureDay(
        api: APIClient,
        cameras: [SyncGridCamera],
        dayStart: Double,
        newestCoverageEnd: Double?,
        playhead target: Double
    ) {
        self.api = api
        self.dayStart = dayStart
        self.dayEnd = dayStart + TimelineTime.day
        self.newestCoverageEnd = newestCoverageEnd

        rebuildModels(to: cameras) { model, cam in
            model.configure(api: api, segments: cam.segments, dayStart: dayStart)
        }

        isPlaying = false
        isScrubbing = false
        playhead = TimelineTime.clamp(target, dayStart, dayEnd - 1)
        for m in models { m.applyRate(rate); m.applyMute(muted) }
        updateLeader()
        fanSeek(to: playhead, shouldPlay: false)
    }

    /// Diff update when the on-view set changes (same day): keep the existing,
    /// still-playing players untouched; configure + align only the newly-added
    /// cameras to the current shared moment; drop the removed ones.
    func setOnView(api: APIClient, cameras: [SyncGridCamera], dayStart: Double) {
        self.api = api
        self.dayStart = dayStart
        self.dayEnd = dayStart + TimelineTime.day

        rebuildModels(to: cameras) { model, cam in
            model.configure(api: api, segments: cam.segments, dayStart: dayStart)
            model.applyRate(rate)
            model.applyMute(muted)
            model.seek(toWall: playhead, shouldPlay: isPlaying)
        } configureExisting: { _, _ in }
        updateLeader()
    }

    /// No cameras on view / no footage — tear every player down.
    func clear() {
        for m in models { m.clear() }
        models = []
        leaderCamera = nil
        isPlaying = false
    }

    /// Reconcile `models` to `cameras` (order + membership). `configureNew` runs
    /// for cameras without an existing player; `configureExisting` for reused
    /// ones (defaults to a full reconfigure — used on a day rebuild).
    private func rebuildModels(
        to cameras: [SyncGridCamera],
        configureNew: (SyncCameraPlayerModel, SyncGridCamera) -> Void,
        configureExisting: ((SyncCameraPlayerModel, SyncGridCamera) -> Void)? = nil
    ) {
        var byName: [String: SyncCameraPlayerModel] = [:]
        for m in models { byName[m.camera] = m }

        var next: [SyncCameraPlayerModel] = []
        var keep = Set<String>()
        for cam in cameras {
            keep.insert(cam.camera)
            if let existing = byName[cam.camera] {
                existing.friendlyName = cam.friendlyName
                if let configureExisting {
                    configureExisting(existing, cam)
                } else {
                    configureNew(existing, cam)
                }
                next.append(existing)
            } else {
                let model = SyncCameraPlayerModel(camera: cam.camera, friendlyName: cam.friendlyName)
                configureNew(model, cam)
                next.append(model)
            }
        }
        for m in models where !keep.contains(m.camera) { m.clear() }
        models = next
    }

    // MARK: Transport

    func togglePlay() { isPlaying ? pause() : play() }

    func play() {
        guard !models.isEmpty else { return }
        isPlaying = true
        for m in models { m.applyPlaying(true) }
        updateLeader()
    }

    func pause() {
        isPlaying = false
        for m in models { m.applyPlaying(false) }
    }

    func setRate(_ r: Float) {
        rate = r
        for m in models { m.applyRate(r) }
    }

    func setMuted(_ m: Bool) {
        muted = m
        for model in models { model.applyMute(m) }
    }

    func toggleMute() { setMuted(!muted) }

    /// Live-drag update: park the displayed playhead without seeking video.
    func scrub(to t: Double) {
        playhead = TimelineTime.clamp(t, dayStart, dayEnd - 1)
    }

    /// Explicit seek to a wall-clock moment — fans out to EVERY player so they
    /// re-align to the same instant. Adopts the current playing state.
    func seek(toWall t: Double) {
        isScrubbing = false
        fanSeek(to: t, shouldPlay: isPlaying)
    }

    func skip(by delta: Double) {
        seek(toWall: playhead + delta)
    }

    func jumpToNewest() {
        guard let end = newestCoverageEnd else { return }
        seek(toWall: TimelineTime.clamp(end - 2, dayStart, dayEnd - 1))
    }

    var canJumpToNewest: Bool { newestCoverageEnd != nil }

    private func fanSeek(to t: Double, shouldPlay: Bool) {
        let target = TimelineTime.clamp(t, dayStart, dayEnd - 1)
        playhead = target
        for m in models { m.seek(toWall: target, shouldPlay: shouldPlay) }
        updateLeader()
    }

    // MARK: Leader + follow

    /// Leader = the first on-view camera WITH footage in the hour window at the
    /// playhead (so a footage-less first tile never freezes the group), falling
    /// back to the first on-view camera. Only the leader reports its position.
    private func updateLeader() {
        let leader = models.first(where: { $0.hasFootage(inHourWindowContaining: playhead) })
            ?? models.first
        for m in models {
            let isLeader = (m === leader)
            m.setLeader(isLeader)
            if isLeader {
                m.onLeaderFollow = { [weak self] wall in self?.follow(wall) }
                m.onLeaderRollover = { [weak self] wall in self?.leaderRolledOver(toWall: wall) }
            }
        }
        leaderCamera = leader?.camera
    }

    /// Leader time observer -> shared playhead (only while actively playing and
    /// not scrubbing; ignores stray ticks from seeks).
    private func follow(_ t: Double) {
        guard isPlaying, !isScrubbing else { return }
        playhead = TimelineTime.clamp(t, dayStart, dayEnd - 1)
    }

    /// The leader crossed an hour boundary into the next covered hour — re-align
    /// EVERY player (the leader included) so they roll into the next hour
    /// together. The leader's bounded hour-window item has ended and paused
    /// (actionAtItemEnd = .pause), so it must reload its own next window here too
    /// — otherwise it would freeze at the old boundary and, as the playhead
    /// driver, stall the whole group.
    private func leaderRolledOver(toWall t: Double) {
        let target = TimelineTime.clamp(t, dayStart, dayEnd - 1)
        playhead = target
        for m in models {
            m.seek(toWall: target, shouldPlay: isPlaying)
        }
        updateLeader()
    }

    /// Periodic drift correction: nudge every follower back toward the leader's
    /// playhead when it has drifted past the threshold. No global seek, no
    /// interruption to the leader or the other tiles.
    private func driftTick() {
        guard isPlaying, !isScrubbing else { return }
        let leaderName = leaderCamera
        for m in models where m.camera != leaderName {
            m.driftCorrect(towardWall: playhead, threshold: driftThreshold)
        }
    }
}
