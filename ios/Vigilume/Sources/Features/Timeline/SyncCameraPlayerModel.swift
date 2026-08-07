import AVFoundation
import Combine
import Foundation

/// One camera's synchronized recording player for the multi-camera Timeline —
/// a single-camera refactor of the old TimelinePlayerModel, driven by the
/// TimelineSyncCoordinator instead of owning the transport itself.
///
/// It owns ONE AVPlayer and the same bounded (~1 h, hour-aligned) HLS VOD window
/// logic, but scoped to THIS camera's segments: every explicit seek maps the
/// shared wall-clock playhead through this camera's REAL segments (coverage gaps
/// map correctly), reloading the hour window when the target falls outside it.
/// A window with no footage renders no item (the tile shows a placeholder while
/// the rest of the group keeps playing). Only the coordinator-designated leader
/// installs a periodic observer that reports its playback position back up so the
/// shared playhead follows during play; followers just play their own footage,
/// and the coordinator's drift check + the next explicit seek re-align everyone.
@MainActor
final class SyncCameraPlayerModel: ObservableObject, Identifiable {
    let player = AVPlayer()

    let camera: String
    var friendlyName: String

    nonisolated var id: String { camera }

    /// True when the loaded hour window has footage (the tile shows video).
    @Published private(set) var hasWindow = false

    private var api: APIClient?
    private var segments: [RecordingSegment] = []
    private var dayStart: Double = 0
    private var dayEnd: Double = 0
    private var window: (start: Double, end: Double)?

    // Shared transport state, mirrored locally so a window (re)load can re-adopt
    // it in the item-ready handler (a fresh AVPlayerItem resets rate/mute).
    private var sharedPlaying = false
    private var sharedRate: Float = 1
    private var sharedMuted = true

    // Pending work applied once the freshly-loaded item becomes ready to play.
    private var pendingSeekMedia: Double?
    private var pendingPlay = false

    // Leader wiring (installed only while this model is the leader).
    private(set) var isLeader = false
    var onLeaderFollow: ((Double) -> Void)?
    var onLeaderRollover: ((Double) -> Void)?

    private var timeObserver: Any?
    private var itemEndObserver: NSObjectProtocol?
    private var statusObservation: NSKeyValueObservation?

    init(camera: String, friendlyName: String) {
        self.camera = camera
        self.friendlyName = friendlyName
        player.actionAtItemEnd = .pause
        player.isMuted = sharedMuted
    }

    deinit {
        if let timeObserver { player.removeTimeObserver(timeObserver) }
        if let itemEndObserver { NotificationCenter.default.removeObserver(itemEndObserver) }
        statusObservation?.invalidate()
    }

    // MARK: Configuration

    /// Point the player at this camera's day of footage. Does NOT seek/play on
    /// its own — the coordinator issues the aligning seek.
    func configure(
        api: APIClient,
        segments: [RecordingSegment],
        dayStart: Double
    ) {
        self.api = api
        self.segments = segments.sorted { $0.start < $1.start }
        self.dayStart = dayStart
        self.dayEnd = dayStart + TimelineTime.day
        window = nil
        hasWindow = false
        pendingSeekMedia = nil
        pendingPlay = false
        replaceItem(nil)
    }

    /// Drop everything (camera removed from view / no footage for the day).
    func clear() {
        player.pause()
        api = nil
        segments = []
        window = nil
        hasWindow = false
        pendingSeekMedia = nil
        pendingPlay = false
        setLeader(false)
        replaceItem(nil)
    }

    // MARK: Seek (driven by the coordinator's shared wall-clock target)

    /// Seek this player to a shared wall-clock instant, reloading the hour window
    /// when the target falls outside the loaded one. `shouldPlay` resumes once
    /// the seek lands (adopted through a window reload via the ready handler).
    func seek(toWall t: Double, shouldPlay: Bool) {
        guard api != nil else { return }
        let target = TimelineTime.clamp(t, dayStart, dayEnd - 1)
        if let window, target >= window.start, target < window.end,
           player.currentItem != nil, hasWindow {
            let mediaTime = TimelineTime.mediaTime(forWall: target, segments: segments, window: window)
            player.seek(
                to: CMTime(seconds: mediaTime, preferredTimescale: 600),
                toleranceBefore: .zero, toleranceAfter: .zero
            )
            if shouldPlay { resume() }
        } else {
            loadWindow(containing: target, seekTo: target, thenPlay: shouldPlay)
        }
    }

    /// Lightweight single-player correction toward the leader's wall-clock
    /// playhead — no window reload, only if the target is inside the loaded,
    /// covered window and the drift exceeds the threshold. Keeps followers
    /// locked sub-second without interrupting the rest of the group.
    func driftCorrect(towardWall target: Double, threshold: Double = 0.75) {
        guard hasWindow, let window, player.currentItem != nil, sharedPlaying,
              target >= window.start, target < window.end else { return }
        let cur = player.currentTime().seconds
        guard cur.isFinite else { return }
        let curWall = TimelineTime.wallTime(forMedia: cur, segments: segments, window: window)
        guard abs(curWall - target) > threshold else { return }
        let mediaTime = TimelineTime.mediaTime(forWall: target, segments: segments, window: window)
        player.seek(
            to: CMTime(seconds: mediaTime, preferredTimescale: 600),
            toleranceBefore: .zero, toleranceAfter: CMTime(seconds: 0.25, preferredTimescale: 600)
        )
    }

    // MARK: Shared transport application

    func applyPlaying(_ playing: Bool) {
        sharedPlaying = playing
        if playing {
            if player.currentItem != nil { resume() }
        } else {
            player.pause()
        }
    }

    func applyRate(_ rate: Float) {
        sharedRate = rate
        player.defaultRate = rate
        if sharedPlaying, player.timeControlStatus != .paused {
            player.rate = rate
        }
    }

    func applyMute(_ muted: Bool) {
        sharedMuted = muted
        player.isMuted = muted
    }

    // MARK: Leader

    /// Whether this camera has footage in the hour window containing `t`
    /// (drives the coordinator's leader election — a footage-less first tile
    /// must never freeze the group).
    func hasFootage(inHourWindowContaining t: Double) -> Bool {
        guard dayEnd > dayStart else { return false }
        let win = TimelineTime.hourWindow(containing: t, dayStart: dayStart, dayEnd: dayEnd)
        return !TimelineTime.segments(segments, inWindow: win).isEmpty
    }

    func setLeader(_ leader: Bool) {
        guard leader != isLeader else { return }
        isLeader = leader
        if leader {
            installTimeObserver()
        } else {
            removeTimeObserver()
            onLeaderFollow = nil
            onLeaderRollover = nil
        }
    }

    // MARK: Window management

    private func loadWindow(containing t: Double, seekTo target: Double, thenPlay: Bool) {
        guard let api else { return }
        let win = TimelineTime.hourWindow(containing: t, dayStart: dayStart, dayEnd: dayEnd)
        let covered = !TimelineTime.segments(segments, inWindow: win).isEmpty
        guard covered else {
            // No footage in this hour: no item, show the placeholder overlay.
            window = nil
            hasWindow = false
            pendingSeekMedia = nil
            pendingPlay = false
            replaceItem(nil)
            player.pause()
            return
        }
        window = win
        hasWindow = true
        pendingSeekMedia = TimelineTime.mediaTime(forWall: target, segments: segments, window: win)
        pendingPlay = thenPlay
        let url = api.recordingPlaylistURL(camera: camera, start: win.start, end: win.end)
        replaceItem(AVPlayerItem(url: url))
    }

    private func replaceItem(_ item: AVPlayerItem?) {
        if let observer = itemEndObserver {
            NotificationCenter.default.removeObserver(observer)
            itemEndObserver = nil
        }
        statusObservation?.invalidate()
        statusObservation = nil

        player.replaceCurrentItem(with: item)
        guard let item else { return }

        itemEndObserver = NotificationCenter.default.addObserver(
            forName: AVPlayerItem.didPlayToEndTimeNotification,
            object: item, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated { self?.handleWindowEnded() }
        }
        statusObservation = item.observe(\.status, options: [.new]) { [weak self] observedItem, _ in
            guard observedItem.status == .readyToPlay else { return }
            MainActor.assumeIsolated { self?.handleReady() }
        }
    }

    /// A (re)loaded window reset the AVPlayer's rate/mute — re-apply the shared
    /// transport, seek to the pending media time, and resume if the group is
    /// playing (or an explicit seek / auto-advance asked to play). Mirrors the
    /// web SyncPlayer's handleReady.
    private func handleReady() {
        if let mt = pendingSeekMedia {
            player.seek(
                to: CMTime(seconds: mt, preferredTimescale: 600),
                toleranceBefore: .zero, toleranceAfter: .zero
            )
            pendingSeekMedia = nil
        }
        player.isMuted = sharedMuted
        player.defaultRate = sharedRate
        if pendingPlay || sharedPlaying {
            resume()
        }
        pendingPlay = false
    }

    /// The hour window played out — roll into the next covered hour, if any.
    /// The leader additionally notifies the coordinator so followers roll into
    /// the next hour together rather than one-at-a-time.
    private func handleWindowEnded() {
        guard let window else { return }
        if let next = segments.first(where: { $0.start >= window.end }), next.start < dayEnd {
            if isLeader {
                onLeaderRollover?(next.start)
            } else {
                loadWindow(containing: next.start, seekTo: next.start, thenPlay: sharedPlaying)
            }
        } else {
            player.pause()
        }
    }

    private func resume() {
        player.defaultRate = sharedRate
        player.playImmediately(atRate: sharedRate)
    }

    private func installTimeObserver() {
        guard timeObserver == nil else { return }
        let interval = CMTime(seconds: 0.5, preferredTimescale: 600)
        timeObserver = player.addPeriodicTimeObserver(
            forInterval: interval, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated { self?.followPlayback() }
        }
    }

    private func removeTimeObserver() {
        if let timeObserver {
            player.removeTimeObserver(timeObserver)
            self.timeObserver = nil
        }
    }

    /// Leader-only: map media time -> wall clock and report it up so the shared
    /// playhead follows. Ignores stray ticks during a pending seek.
    private func followPlayback() {
        guard isLeader, let window, pendingSeekMedia == nil,
              player.timeControlStatus == .playing else { return }
        let media = player.currentTime().seconds
        guard media.isFinite else { return }
        let wall = TimelineTime.wallTime(forMedia: media, segments: segments, window: window)
        onLeaderFollow?(wall)
    }
}
