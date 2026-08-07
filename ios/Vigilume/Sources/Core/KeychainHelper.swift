import Foundation
import Security

/// Minimal generic-password Keychain wrapper. JWTs are stored here, never in
/// UserDefaults (docs/ios-design.md §1).
enum KeychainHelper {
    private static let service = "com.vigilume.app"
    /// Pre-rename service (the app shipped as "Sentinel"). Read-only: items
    /// found here are moved to `service` on first read and then deleted, so an
    /// existing install keeps its session instead of being logged out by the
    /// rename. Safe to delete once no install can still be on a pre-rename build.
    private static let legacyService = "com.sentinelnvr.app"

    static func setString(_ value: String, forKey key: String) {
        let data = Data(value.utf8)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: key,
        ]
        let attributes: [String: Any] = [
            kSecValueData as String: data,
            // Available after first unlock so a push-triggered launch can read it.
            kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlock,
        ]
        let status = SecItemUpdate(query as CFDictionary, attributes as CFDictionary)
        if status == errSecItemNotFound {
            var add = query
            add.merge(attributes) { _, new in new }
            SecItemAdd(add as CFDictionary, nil)
        }
    }

    /// Read `key`, transparently adopting a value left behind under the
    /// pre-rename service.
    ///
    /// The migration is lazy (per key, on first read), idempotent — once the
    /// item lives under `service` the legacy lookup never runs again — and a
    /// no-op when nothing was stored. The legacy copy is deleted only after the
    /// new one reads back successfully, so a failed write leaves the original
    /// in place to retry rather than destroying the token.
    static func string(forKey key: String) -> String? {
        if let value = read(key, from: service) { return value }
        guard let legacy = read(key, from: legacyService) else { return nil }
        setString(legacy, forKey: key)
        guard read(key, from: service) != nil else { return legacy }
        delete(key, from: legacyService)
        return legacy
    }

    static func remove(forKey key: String) {
        delete(key, from: service)
        // Also drop any not-yet-migrated copy, so removing a server really
        // forgets its token instead of leaving one to be re-adopted later.
        delete(key, from: legacyService)
    }

    private static func read(_ key: String, from service: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: key,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var result: AnyObject?
        guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess,
              let data = result as? Data
        else { return nil }
        return String(data: data, encoding: .utf8)
    }

    private static func delete(_ key: String, from service: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: key,
        ]
        SecItemDelete(query as CFDictionary)
    }
}
