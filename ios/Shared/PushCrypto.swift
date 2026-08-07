import CryptoKit
import Foundation
import Security

/// E2E push crypto + per-server key storage, shared by the app target AND the
/// NotificationService extension (docs/push-architecture.md §2).
///
/// Scheme (pinned contract):
/// - At registration the app generates a random 32-byte key per server and
///   sends `key_b64` to that hoster server; the key never touches the relay.
/// - A push arrives as `userInfo["enc"] = base64(nonce12 || ciphertext ||
///   tag16)`, AES-256-GCM, no AAD — exactly CryptoKit's
///   `AES.GCM.SealedBox.combined` layout, so decrypt is
///   `AES.GCM.open(SealedBox(combined:), using: key)`.
/// - Plaintext is UTF-8 JSON `{title, body, event_id, snapshot_url|null}`.
///
/// Keys live in the Keychain under a dedicated service so the extension can
/// enumerate ONLY push keys (never session JWTs, which use a different
/// service). Both targets share the `com.vigilume.app.shared` keychain
/// access group (see project.yml); it is the only group listed in each
/// target's entitlements, so it is the DEFAULT write group — items are added
/// without an explicit `kSecAttrAccessGroup`, which also keeps this file
/// compilable for the standalone check script (Scripts/PushCryptoCheck.swift).
enum PushCrypto {

    // MARK: Payload

    /// Decrypted notification payload (plaintext JSON is snake_case).
    ///
    /// `camera` (slug) and `cameraLabel` (friendly name) are OPTIONAL: newer
    /// hoster servers add them so the NotificationService extension can group
    /// same-camera notifications (thread id = `camera`, group summary =
    /// `cameraLabel`). Older servers omit them and decode leaves both nil —
    /// the extension falls back to a constant thread id. The explicit
    /// initializer defaults them to nil so existing call sites (the check
    /// script's round-trip vectors) keep compiling unchanged.
    struct Payload: Equatable {
        let title: String
        let body: String
        let eventID: String
        let snapshotURL: String?
        let camera: String?
        let cameraLabel: String?

        init(
            title: String,
            body: String,
            eventID: String,
            snapshotURL: String?,
            camera: String? = nil,
            cameraLabel: String? = nil
        ) {
            self.title = title
            self.body = body
            self.eventID = eventID
            self.snapshotURL = snapshotURL
            self.camera = camera
            self.cameraLabel = cameraLabel
        }
    }

    enum PushCryptoError: Error {
        case badBase64
        case tooShort
        case badJSON
    }

    // MARK: Decrypt / encrypt

    /// Open `base64(nonce12 || ciphertext || tag16)` with `key` and parse the
    /// plaintext JSON. Throws on any malformed input or GCM auth failure
    /// (wrong key / tampered ciphertext).
    static func decrypt(_ encB64: String, using key: SymmetricKey) throws -> Payload {
        guard let combined = Data(base64Encoded: encB64) else {
            throw PushCryptoError.badBase64
        }
        // SealedBox(combined:) needs at least nonce(12) + tag(16) + 1 byte.
        guard combined.count > 12 + 16 else { throw PushCryptoError.tooShort }
        let box = try AES.GCM.SealedBox(combined: combined)
        let plaintext = try AES.GCM.open(box, using: key)
        return try parse(plaintext)
    }

    /// Try every stored per-server key (small N; the GCM tag safely rejects
    /// wrong keys). Returns the payload plus WHICH server's key matched —
    /// pushes don't identify their sender, this is how the extension finds
    /// out. Never throws: any failure means "not decryptable" (nil).
    static func decryptWithAnyStoredKey(_ encB64: String)
        -> (serverID: String, payload: Payload)?
    {
        for entry in allKeys() {
            if let payload = try? decrypt(encB64, using: entry.key) {
                return (entry.serverID, payload)
            }
        }
        return nil
    }

    /// Produce the wire format (`base64(nonce12 || ciphertext || tag16)`).
    /// The app never sends pushes — this exists for the debug check script
    /// (round-trip assertion) and mirrors the hoster server's encrypt side.
    static func seal(
        _ payload: Payload, using key: SymmetricKey, nonce: AES.GCM.Nonce? = nil
    ) throws -> String {
        var json: [String: Any] = [
            "title": payload.title,
            "body": payload.body,
            "event_id": payload.eventID,
        ]
        json["snapshot_url"] = payload.snapshotURL ?? NSNull()
        // camera/camera_label are optional — only present when the server sent
        // them (mirrors the hoster's build_plaintext, which omits absent keys).
        if let camera = payload.camera { json["camera"] = camera }
        if let cameraLabel = payload.cameraLabel { json["camera_label"] = cameraLabel }
        let plaintext = try JSONSerialization.data(withJSONObject: json)
        let box = try AES.GCM.seal(plaintext, using: key, nonce: nonce)
        guard let combined = box.combined else { throw PushCryptoError.tooShort }
        return combined.base64EncodedString()
    }

    private static func parse(_ plaintext: Data) throws -> Payload {
        guard
            let obj = try? JSONSerialization.jsonObject(with: plaintext) as? [String: Any],
            let title = obj["title"] as? String,
            let body = obj["body"] as? String
        else { throw PushCryptoError.badJSON }
        // event_id is a string per contract; tolerate a number.
        let eventID: String
        if let s = obj["event_id"] as? String {
            eventID = s
        } else if let n = obj["event_id"] as? NSNumber {
            eventID = n.stringValue
        } else {
            throw PushCryptoError.badJSON
        }
        let snapshotURL = obj["snapshot_url"] as? String   // nil for JSON null
        // Optional grouping fields — nil when absent (older servers) or null.
        let camera = obj["camera"] as? String
        let cameraLabel = obj["camera_label"] as? String
        return Payload(
            title: title, body: body, eventID: eventID, snapshotURL: snapshotURL,
            camera: camera, cameraLabel: cameraLabel
        )
    }

    // MARK: Keychain (per-server key storage)

    /// Dedicated service: enumerating it returns push keys only.
    private static let service = "com.vigilume.push"
    /// Pre-rename service (the app shipped as "Sentinel"). Keys stored here by
    /// an older build are moved to `service` before every read — losing them
    /// would leave already-registered servers sending pushes this device can no
    /// longer decrypt, with no re-registration to fix it.
    private static let legacyService = "com.sentinelnvr.push"

    /// Move any pre-rename keys onto the current service, then forget them.
    ///
    /// Idempotent and cheap in the steady state: one keychain query that misses
    /// once the legacy service is empty (or was never used). A key is deleted
    /// from the legacy service only after it reads back under the new one, so a
    /// partial failure is retried on the next call instead of losing the key.
    /// Both targets run this; whichever touches the keychain first wins, and a
    /// concurrent second pass is a no-op.
    private static func migrateLegacyKeys() {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: legacyService,
            kSecMatchLimit as String: kSecMatchLimitAll,
            kSecReturnAttributes as String: true,
            kSecReturnData as String: true,
        ]
        var result: AnyObject?
        guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess,
              let items = result as? [[String: Any]]
        else { return }
        for item in items {
            guard let account = item[kSecAttrAccount as String] as? String,
                  let data = item[kSecValueData as String] as? Data,
                  data.count == 32
            else { continue }
            if rawKey(forServer: account, in: service) == nil {
                store(key: SymmetricKey(data: data), forServer: account)
                guard rawKey(forServer: account, in: service) != nil else { continue }
            }
            deleteKey(forServer: account, in: legacyService)
        }
    }

    /// Return the stored key for a server, or nil if none exists yet.
    static func storedKey(forServer serverID: String) -> SymmetricKey? {
        migrateLegacyKeys()
        return rawKey(forServer: serverID, in: service)
    }

    private static func rawKey(forServer serverID: String, in service: String) -> SymmetricKey? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: serverID,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var result: AnyObject?
        guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess,
              let data = result as? Data, data.count == 32
        else { return nil }
        return SymmetricKey(data: data)
    }

    /// Return the server's key, generating and persisting a fresh 32-byte one
    /// only if absent (docs/push-architecture.md: key rotates only when the
    /// registration is recreated from scratch, not on every re-register).
    static func ensureKey(forServer serverID: String) -> SymmetricKey {
        if let existing = storedKey(forServer: serverID) { return existing }
        let key = SymmetricKey(size: .bits256)
        store(key: key, forServer: serverID)
        return key
    }

    static func removeKey(forServer serverID: String) {
        deleteKey(forServer: serverID, in: service)
        // Drop any not-yet-migrated copy too, or the next read would resurrect
        // the key for a server that has just been removed.
        deleteKey(forServer: serverID, in: legacyService)
    }

    private static func deleteKey(forServer serverID: String, in service: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: serverID,
        ]
        SecItemDelete(query as CFDictionary)
    }

    /// Every stored (serverID, key) pair — the extension iterates these to
    /// find the sender.
    static func allKeys() -> [(serverID: String, key: SymmetricKey)] {
        migrateLegacyKeys()
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecMatchLimit as String: kSecMatchLimitAll,
            kSecReturnAttributes as String: true,
            kSecReturnData as String: true,
        ]
        var result: AnyObject?
        guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess,
              let items = result as? [[String: Any]]
        else { return [] }
        return items.compactMap { item in
            guard let account = item[kSecAttrAccount as String] as? String,
                  let data = item[kSecValueData as String] as? Data,
                  data.count == 32
            else { return nil }
            return (account, SymmetricKey(data: data))
        }
    }

    /// Base64 of the server's key for the register body (`key_b64`).
    static func keyB64(_ key: SymmetricKey) -> String {
        key.withUnsafeBytes { Data($0) }.base64EncodedString()
    }

    private static func store(key: SymmetricKey, forServer serverID: String) {
        let data = key.withUnsafeBytes { Data($0) }
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: serverID,
        ]
        let attributes: [String: Any] = [
            kSecValueData as String: data,
            // The extension must read this while the phone is locked
            // (a push can arrive any time after first unlock).
            kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlock,
        ]
        let status = SecItemUpdate(query as CFDictionary, attributes as CFDictionary)
        if status == errSecItemNotFound {
            var add = query
            add.merge(attributes) { _, new in new }
            SecItemAdd(add as CFDictionary, nil)
        }
    }
}
