import UserNotifications

/// Runs on `mutable-content: 1` pushes (docs/push-architecture.md §2).
///
/// E2E path: the push arrives as a generic alert ("Vigilume / Encrypted
/// notification") plus `userInfo["enc"] = base64(nonce12||ciphertext||tag16)`
/// (AES-256-GCM). The push does NOT say which server sent it, so we try every
/// per-server key stored in the shared Keychain group — the GCM tag safely
/// rejects wrong keys. On success we replace title/body, enrich userInfo with
/// `event_id` + `server_id` for tap routing, and fetch + attach the snapshot
/// (`snapshot_url` is an absolute, self-tokened URL on the HOSTER server —
/// never the relay) with a short timeout. ANY failure delivers the generic
/// alert unchanged; this extension must never crash (a crash shows the raw
/// fallback anyway, but burns the ~30 s budget).
///
/// Legacy path: plaintext pushes that carry `snapshot_url` directly in
/// userInfo (pre-E2E servers) still get the image attached.
final class NotificationService: UNNotificationServiceExtension {
    private var handler: ((UNNotificationContent) -> Void)?
    /// Best deliverable so far: decrypted-but-unattached beats the raw push.
    private var bestContent: UNNotificationContent?

    /// Snapshot fetches get a short leash (~4 s per the contract) so a slow
    /// or unreachable hoster still leaves time to deliver the decrypted text.
    private static let snapshotSession: URLSession = {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 4
        config.timeoutIntervalForResource = 8
        return URLSession(configuration: config)
    }()

    override func didReceive(
        _ request: UNNotificationRequest,
        withContentHandler contentHandler: @escaping (UNNotificationContent) -> Void
    ) {
        handler = contentHandler
        bestContent = request.content

        guard let content = request.content.mutableCopy() as? UNMutableNotificationContent
        else {
            contentHandler(request.content)
            return
        }

        var snapshotURL: URL?

        if let enc = request.content.userInfo["enc"] as? String {
            // E2E push: decrypt or deliver the generic alert unchanged.
            guard let match = PushCrypto.decryptWithAnyStoredKey(enc) else {
                contentHandler(request.content)
                return
            }
            content.title = match.payload.title
            content.body = match.payload.body
            // Per-camera grouping (docs/push-architecture.md §2): iOS stacks
            // notifications that share a threadIdentifier and lets the user
            // expand + tap through each one. The camera slug travels inside
            // the ciphertext; when absent (older servers) fall back to a
            // single constant thread so nothing is lost.
            //
            // No summaryArgument: it was deprecated in iOS 15 and is IGNORED at
            // runtime, so setting it only produced a warning. iOS now names the
            // collapsed group itself, and there is no replacement API — the
            // payload still carries `cameraLabel` for any future use.
            content.threadIdentifier = match.payload.camera ?? "vigilume"
            // Carry the decrypted event + the server whose key matched into
            // the tap-routing path (AppDelegate.didReceive reads these). Each
            // notification keeps its OWN event_id, so tapping any single one in
            // an expanded group deep-links to that one's event.
            var userInfo = content.userInfo
            userInfo["event_id"] = match.payload.eventID
            userInfo["server_id"] = match.serverID
            content.userInfo = userInfo
            if let s = match.payload.snapshotURL { snapshotURL = URL(string: s) }
        } else if let s = request.content.userInfo["snapshot_url"] as? String {
            // Legacy plaintext push: text is already right, just attach.
            snapshotURL = URL(string: s)
        }

        bestContent = content
        guard let snapshotURL else {
            contentHandler(content)
            return
        }

        Self.snapshotSession.downloadTask(with: snapshotURL) { tmp, _, _ in
            defer { contentHandler(content) }
            guard let tmp else { return }
            // UNNotificationAttachment requires a recognizable file extension.
            let dst = tmp.deletingLastPathComponent()
                .appendingPathComponent(UUID().uuidString + ".jpg")
            do {
                try FileManager.default.moveItem(at: tmp, to: dst)
                let attachment = try UNNotificationAttachment(identifier: "snapshot", url: dst)
                content.attachments = [attachment]
            } catch {
                // Deliver without the image.
            }
        }.resume()
    }

    override func serviceExtensionTimeWillExpire() {
        if let bestContent {
            handler?(bestContent)   // decrypted-if-possible, un-attached
        }
    }
}
