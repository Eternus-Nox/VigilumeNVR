import Foundation
import Photos

/// Downloads an event clip (mp4) or snapshot (jpg) from its tokened URL and
/// adds it to the user's photo library (add-only access —
/// NSPhotoLibraryAddUsageDescription). One instance drives one save button:
/// `phase` feeds the button's progress / success / failure states.
@MainActor
final class MediaSaver: ObservableObject {
    enum Kind {
        case video
        case photo
    }

    enum Phase: Equatable {
        case idle
        case requestingAccess
        /// Download running; fraction is nil until the server sends a length.
        case downloading(Double?)
        case saving
        case saved
        case failed(String)
    }

    @Published private(set) var phase: Phase = .idle

    var isBusy: Bool {
        switch phase {
        case .requestingAccess, .downloading, .saving: return true
        case .idle, .saved, .failed: return false
        }
    }

    private var saveTask: Task<Void, Never>?

    func save(from url: URL, kind: Kind) {
        guard !isBusy else { return }
        saveTask = Task { await run(url: url, kind: kind) }
    }

    private func run(url: URL, kind: Kind) async {
        phase = .requestingAccess
        let status = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
        guard status == .authorized || status == .limited else {
            phase = .failed(
                "Photos access is off — allow \"Add Photos Only\" for Vigilume in Settings."
            )
            return
        }

        do {
            phase = .downloading(nil)
            let downloader = ProgressDownloader()
            let fileURL = try await downloader.download(
                from: url,
                fileExtension: kind == .video ? "mp4" : "jpg"
            ) { [weak self] fraction in
                Task { @MainActor [weak self] in
                    guard let self, case .downloading = self.phase else { return }
                    self.phase = .downloading(fraction)
                }
            }
            defer { try? FileManager.default.removeItem(at: fileURL) }

            phase = .saving
            try await PHPhotoLibrary.shared().performChanges {
                switch kind {
                case .video:
                    PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: fileURL)
                case .photo:
                    PHAssetChangeRequest.creationRequestForAssetFromImage(atFileURL: fileURL)
                }
            }
            phase = .saved
            scheduleIdleReset()
        } catch let error as MediaSaveError {
            phase = .failed(error.message)
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    /// Let the "Saved" checkmark linger, then offer the button again.
    private func scheduleIdleReset() {
        Task { [weak self] in
            try? await Task.sleep(for: .seconds(4))
            guard let self, self.phase == .saved else { return }
            self.phase = .idle
        }
    }
}

struct MediaSaveError: Error {
    let message: String
}

/// Delegate-based URLSession download with byte-level progress, streaming to
/// a temp file (event clips can be tens of MB — never buffered in memory).
private final class ProgressDownloader: NSObject, URLSessionDownloadDelegate {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<URL, Error>?
    private var onProgress: ((Double) -> Void)?
    private var destination: URL?
    private var session: URLSession?

    func download(
        from url: URL,
        fileExtension: String,
        onProgress: @escaping (Double) -> Void
    ) async throws -> URL {
        self.onProgress = onProgress
        destination = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension(fileExtension)

        let session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
        self.session = session

        return try await withCheckedThrowingContinuation { cont in
            lock.lock()
            continuation = cont
            lock.unlock()
            session.downloadTask(with: url).resume()
        }
    }

    // MARK: URLSessionDownloadDelegate (background queue)

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        guard totalBytesExpectedToWrite > 0 else { return }
        onProgress?(Double(totalBytesWritten) / Double(totalBytesExpectedToWrite))
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        // The temp file only lives for the duration of this callback — move
        // it synchronously to our own destination with the right extension.
        do {
            if let http = downloadTask.response as? HTTPURLResponse,
               !(200 ..< 300).contains(http.statusCode) {
                throw MediaSaveError(
                    message: http.statusCode == 401 || http.statusCode == 403
                        ? "The download was rejected — sign in again and retry."
                        : "The server returned HTTP \(http.statusCode) for this file."
                )
            }
            guard let destination else {
                throw MediaSaveError(message: "Internal error: missing download destination.")
            }
            try? FileManager.default.removeItem(at: destination)
            try FileManager.default.moveItem(at: location, to: destination)
            finish(.success(destination))
        } catch {
            finish(.failure(error))
        }
    }

    func urlSession(
        _ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?
    ) {
        if let error {
            finish(.failure(error))
        }
        // Success already resumed in didFinishDownloadingTo; either way the
        // task is over — break the session→delegate retain cycle.
        session.finishTasksAndInvalidate()
    }

    private func finish(_ result: Result<URL, Error>) {
        lock.lock()
        let cont = continuation
        continuation = nil
        lock.unlock()
        guard let cont else { return }
        switch result {
        case .success(let url): cont.resume(returning: url)
        case .failure(let error): cont.resume(throwing: error)
        }
    }
}
