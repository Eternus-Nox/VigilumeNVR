import Combine
import SwiftUI
import UserNotifications

/// Owns the APNs push-registration lifecycle for this device:
/// authorization prompt → `registerForRemoteNotifications()` → hex token from
/// the AppDelegate → POST /api/notifications/apns/register
/// {device_token, device_name, key_b64, environment}
/// (docs/push-architecture.md §2 — key_b64 is the per-server E2E key).
///
/// The backend route is being built in parallel (docs/ios-design.md §3.1); a
/// 404 is surfaced as `.serverUnsupported` ("server not updated yet") and the
/// app re-registers automatically once the server gains the route. The
/// "push enabled" choice is stored per server, so each saved server keeps its
/// own preference across switches.
@MainActor
final class PushManager: ObservableObject {

    enum Status: Equatable {
        case off                    // push not enabled on this device
        case requestingPermission
        case awaitingToken          // waiting for APNs to hand us a token
        case registering
        case registered
        case permissionDenied       // declined in iOS Settings
        case serverUnsupported      // backend route 404s — server not updated yet
        case failed(String)
    }

    static let shared = PushManager()

    @Published private(set) var status: Status = .off
    /// Whether push is enabled for the ACTIVE server (mirrors the stored flag).
    @Published private(set) var isEnabled = false
    @Published private(set) var authorization: UNAuthorizationStatus = .notDetermined

    private var deviceTokenHex: String?
    private var cancellables = Set<AnyCancellable>()
    private let defaults = UserDefaults.standard

    private init() {
        // Hex token posted by the AppDelegate on every (re)registration.
        NotificationCenter.default.publisher(for: AppDelegate.apnsTokenNotification)
            .compactMap { $0.userInfo?["token"] as? String }
            .sink { [weak self] hex in
                Task { @MainActor in self?.handleDeviceToken(hex) }
            }
            .store(in: &cancellables)

        // APNs tokens can rotate: re-request registration on every activation
        // while push is enabled (ios-design.md §3.2).
        NotificationCenter.default.publisher(for: UIApplication.didBecomeActiveNotification)
            .sink { [weak self] _ in
                Task { @MainActor in self?.appDidBecomeActive() }
            }
            .store(in: &cancellables)

        syncToActiveServer()
    }

    // MARK: User actions

    /// Enable push on this device for the active server: ask permission,
    /// register with APNs, then POST the token to the backend.
    func enable() async {
        status = .requestingPermission
        let granted: Bool
        do {
            granted = try await UNUserNotificationCenter.current()
                .requestAuthorization(options: [.alert, .sound, .badge])
        } catch {
            status = .failed(error.localizedDescription)
            return
        }
        await refreshAuthorization()
        guard granted else {
            setEnabledFlag(false)
            isEnabled = false
            status = .permissionDenied
            return
        }
        setEnabledFlag(true)
        isEnabled = true
        status = .awaitingToken
        UIApplication.shared.registerForRemoteNotifications()
        if let hex = deviceTokenHex {
            await register(tokenHex: hex)
        }
    }

    /// Disable push for the active server (best-effort server unregister).
    func disable() async {
        setEnabledFlag(false)
        isEnabled = false
        status = .off
        guard let hex = deviceTokenHex, let (client, _) = makeClient() else { return }
        try? await client.unregisterAPNs(deviceToken: hex)
    }

    /// Best-effort unregister while the session token is still valid — call
    /// BEFORE SessionModel.logout(). Keeps the per-server enabled flag so the
    /// next login on this server re-registers automatically.
    func unregisterForLogout() async {
        status = .off
        guard isEnabled, let hex = deviceTokenHex, let (client, _) = makeClient() else { return }
        try? await client.unregisterAPNs(deviceToken: hex)
    }

    /// Adopt the (possibly new) active server's stored preference and
    /// re-register against it if enabled. Call after login/server switch.
    func syncToActiveServer() {
        isEnabled = enabledFlag()
        guard isEnabled else {
            status = .off
            return
        }
        if deviceTokenHex == nil { status = .awaitingToken }
        UIApplication.shared.registerForRemoteNotifications()
        if let hex = deviceTokenHex {
            Task { await register(tokenHex: hex) }
        }
    }

    /// Re-read the OS authorization state (the user may have revoked it in
    /// iOS Settings while the app was backgrounded).
    func refreshAuthorization() async {
        let settings = await UNUserNotificationCenter.current().notificationSettings()
        authorization = settings.authorizationStatus
        if isEnabled, authorization == .denied {
            status = .permissionDenied
        }
    }

    // MARK: Internals

    private func appDidBecomeActive() {
        Task { await refreshAuthorization() }
        isEnabled = enabledFlag()
        if isEnabled {
            UIApplication.shared.registerForRemoteNotifications()
        }
    }

    private func handleDeviceToken(_ hex: String) {
        let changed = hex != deviceTokenHex
        deviceTokenHex = hex
        // Re-POST when the token rotated or the last attempt didn't stick
        // (e.g. serverUnsupported — the server may have been updated since).
        guard isEnabled, changed || status != .registered else { return }
        Task { await register(tokenHex: hex) }
    }

    /// APNs environment matching the running binary's `aps-environment`
    /// entitlement: Xcode/debug builds use the sandbox gateway, TestFlight /
    /// App Store builds (auto-signing flips to production) use production.
    /// The backend accepts exactly "sandbox" | "production" (400 otherwise).
    private static var apnsEnvironment: String {
        #if DEBUG
        return "sandbox"
        #else
        return "production"
        #endif
    }

    private func register(tokenHex: String) async {
        guard let (client, serverID) = makeClient() else {
            status = .failed("No signed-in server to register with")
            return
        }
        // Per-server E2E key (docs/push-architecture.md §2): generated once
        // per server, reused on every re-register; the server encrypts pushes
        // with it so the relay only ever sees ciphertext.
        let key = PushCrypto.ensureKey(forServer: serverID.uuidString)
        status = .registering
        do {
            try await client.registerAPNs(
                deviceToken: tokenHex,
                deviceName: String(UIDevice.current.name.prefix(64)),
                keyB64: PushCrypto.keyB64(key),
                environment: Self.apnsEnvironment
            )
            status = .registered
        } catch let error as ApiError where error.status == 404 {
            status = .serverUnsupported
        } catch let error as ApiError {
            status = .failed(error.message)
        } catch {
            status = .failed(error.localizedDescription)
        }
    }

    /// Snapshot client (+ server id, which keys the E2E push key in the
    /// Keychain) for the active server, built fresh so registration works
    /// even from launch paths that never touched the SessionModel
    /// (ServerStore reads are cheap and side-effect free).
    private func makeClient() -> (client: APIClient, serverID: UUID)? {
        let store = ServerStore()
        guard let server = store.activeServer,
              let url = server.url,
              let token = store.activeToken
        else { return nil }
        // APNs registration is control traffic → primary URL only (no media).
        return (APIClient(apiBase: url, token: token), server.id)
    }

    // MARK: Per-server enabled flag

    /// Read the per-server flag, adopting the pre-rename key when the new one
    /// is absent. `bool(forKey:)` reports a missing key as `false`, so skipping
    /// this would silently switch push OFF for every already-enabled server —
    /// invisibly, since the UI would agree it was never enabled. Idempotent
    /// (the legacy key is dropped once copied) and a no-op on a fresh install.
    private func enabledFlag() -> Bool {
        guard let id = ServerStore().activeServerID else { return false }
        let key = Self.enabledKey(id)
        if defaults.object(forKey: key) == nil,
           let legacy = defaults.object(forKey: Self.legacyEnabledKey(id)) as? Bool {
            defaults.set(legacy, forKey: key)
            defaults.removeObject(forKey: Self.legacyEnabledKey(id))
        }
        return defaults.bool(forKey: key)
    }

    private func setEnabledFlag(_ on: Bool) {
        guard let id = ServerStore().activeServerID else { return }
        defaults.set(on, forKey: Self.enabledKey(id))
        // Never let a stale pre-rename value be adopted over an explicit choice.
        defaults.removeObject(forKey: Self.legacyEnabledKey(id))
    }

    private static func enabledKey(_ id: UUID) -> String {
        "vigilume.push.enabled.\(id.uuidString)"
    }

    private static func legacyEnabledKey(_ id: UUID) -> String {
        "sentinel.push.enabled.\(id.uuidString)"
    }
}
