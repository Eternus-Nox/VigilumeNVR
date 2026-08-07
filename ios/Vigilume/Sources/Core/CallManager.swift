import CallKit
import Combine
import PushKit
import UIKit

/// Owns the PushKit VoIP + CallKit "doorbell call" flow: when the AD410 button
/// is pressed the backend sends a VoIP APNs push (apns-push-type: voip, topic
/// <bundle>.voip) and the phone RINGS like a real call. Answering deep-links to
/// the doorbell live view.
///
/// **Why this is separate from `PushManager`.** Alert pushes go through
/// `UNUserNotificationCenter` + the normal APNs device token; VoIP pushes need a
/// dedicated `PKPushRegistry(.voIP)` token registered under a different APNs
/// topic. Both live side-by-side — this class never touches the alert path.
///
/// **The hard iOS rule (iOS 13+):** every VoIP push MUST report a new incoming
/// call to CallKit *synchronously* inside `didReceiveIncomingPushWith` (before
/// its completion handler returns), or the system kills the app and may revoke
/// VoIP privileges. So `reportIncomingCall` is called first, unconditionally.
final class CallManager: NSObject {

    static let shared = CallManager()

    /// Posted (userInfo ["camera": String]) when the user ANSWERS a doorbell
    /// call — the app routes to that camera's live view. The value is the
    /// payload's `camera` (the friendly name, e.g. "Front Door").
    static let openLiveCameraNotification = Notification.Name("VigilumeOpenLiveCamera")

    private let provider: CXProvider
    private var voipRegistry: PKPushRegistry?
    /// Latest VoIP token (hex); resent to the active server on login/switch.
    private var voipTokenHex: String?

    /// The single in-flight doorbell call (we only ever ring one at a time).
    private var activeCallUUID: UUID?
    /// Camera to route to when THIS call is answered (payload `camera`).
    private var activeCallCamera: String?

    private let defaults = UserDefaults.standard

    private override init() {
        let config = CXProviderConfiguration()
        config.supportsVideo = true                 // a doorbell "call" is video
        config.maximumCallGroups = 1
        config.maximumCallsPerCallGroup = 1
        config.supportedHandleTypes = [.generic]
        // Use the ringtone; icon left default (the app icon is applied by iOS).
        provider = CXProvider(configuration: config)
        super.init()
        provider.setDelegate(self, queue: nil)      // nil == main queue

        // The token needs to reach the active server; resend when the app
        // becomes active (covers login, server switch, and cold launch).
        NotificationCenter.default.addObserver(
            self, selector: #selector(appDidBecomeActive),
            name: UIApplication.didBecomeActiveNotification, object: nil
        )
    }

    // MARK: Registration

    /// Register for VoIP pushes. Call once at launch (AppDelegate) — VoIP pushes
    /// must be able to wake the app regardless of login state; the token is only
    /// POSTed to the backend once there's a signed-in server.
    func registerForVoIPPushes() {
        guard voipRegistry == nil else { return }
        let registry = PKPushRegistry(queue: .main)
        registry.delegate = self
        registry.desiredPushTypes = [.voIP]
        voipRegistry = registry
    }

    /// Re-POST the current VoIP token to the (possibly new) active server. Safe
    /// to call repeatedly — a no-op until we hold a token and a signed-in server.
    func syncToActiveServer() {
        guard let hex = voipTokenHex else { return }
        Task { @MainActor in await sendToken(hex) }
    }

    @objc private func appDidBecomeActive() {
        syncToActiveServer()
    }

    @MainActor
    private func sendToken(_ hex: String) async {
        guard let (client, _) = makeClient() else { return }
        do {
            try await client.registerVoIP(
                token: hex,
                deviceName: String(UIDevice.current.name.prefix(64)),
                environment: Self.apnsEnvironment
            )
        } catch {
            // 404 (server predates VoIP) or transient failure — retried on the
            // next activation / server switch via syncToActiveServer().
            NSLog("VoIP token registration failed: \(error.localizedDescription)")
        }
    }

    /// Snapshot client for the active server (mirrors PushManager.makeClient):
    /// VoIP registration is control traffic → primary URL only.
    @MainActor
    private func makeClient() -> (client: APIClient, serverID: UUID)? {
        let store = ServerStore()
        guard let server = store.activeServer,
              let url = server.url,
              let token = store.activeToken
        else { return nil }
        return (APIClient(apiBase: url, token: token), server.id)
    }

    /// APNs environment matching the running binary (sandbox for Xcode/debug,
    /// production for TestFlight/App Store) — same rule as PushManager.
    private static var apnsEnvironment: String {
        #if DEBUG
        return "sandbox"
        #else
        return "production"
        #endif
    }

    // MARK: Incoming call

    /// Report a ringing doorbell call to CallKit. MUST be invoked synchronously
    /// from the VoIP push handler; `completion` is called once CallKit has the
    /// call so the push handler can return.
    func reportIncomingCall(camera: String, eventID: Int?, completion: @escaping () -> Void) {
        // If a stale call is still around (e.g. a missed prior ring), end it so
        // the new ring is the only active call.
        if let old = activeCallUUID {
            provider.reportCall(with: old, endedAt: Date(), reason: .remoteEnded)
        }
        let uuid = UUID()
        activeCallUUID = uuid
        activeCallCamera = camera

        let update = CXCallUpdate()
        update.remoteHandle = CXHandle(type: .generic, value: camera)
        update.localizedCallerName = camera        // "Front Door"
        update.hasVideo = true
        update.supportsHolding = false
        update.supportsGrouping = false
        update.supportsUngrouping = false
        update.supportsDTMF = false

        provider.reportNewIncomingCall(with: uuid, update: update) { error in
            if let error {
                NSLog("reportNewIncomingCall failed: \(error.localizedDescription)")
                // Drop our bookkeeping so a later ring isn't blocked.
                if self.activeCallUUID == uuid {
                    self.activeCallUUID = nil
                    self.activeCallCamera = nil
                }
            }
            completion()
        }
    }

    /// Tear the current CallKit call down (used after answering routes to live,
    /// and on any end/decline).
    private func endActiveCall(reason: CXCallEndedReason? = nil) {
        guard let uuid = activeCallUUID else { return }
        if let reason {
            provider.reportCall(with: uuid, endedAt: Date(), reason: reason)
        }
        activeCallUUID = nil
        activeCallCamera = nil
    }
}

// MARK: - PKPushRegistryDelegate

extension CallManager: PKPushRegistryDelegate {

    func pushRegistry(
        _ registry: PKPushRegistry,
        didUpdate pushCredentials: PKPushCredentials,
        for type: PKPushType
    ) {
        guard type == .voIP else { return }
        let hex = pushCredentials.token.map { String(format: "%02x", $0) }.joined()
        voipTokenHex = hex
        Task { await sendToken(hex) }
    }

    func pushRegistry(_ registry: PKPushRegistry, didInvalidatePushTokenFor type: PKPushType) {
        guard type == .voIP else { return }
        voipTokenHex = nil
    }

    /// The VoIP push arrived. Report the incoming call IMMEDIATELY (hard iOS
    /// requirement) using the camera friendly name from the payload, then let
    /// the completion return once CallKit has it.
    func pushRegistry(
        _ registry: PKPushRegistry,
        didReceiveIncomingPushWith payload: PKPushPayload,
        for type: PKPushType,
        completion: @escaping () -> Void
    ) {
        guard type == .voIP else { completion(); return }

        let dict = payload.dictionaryPayload
        // Backend payload: {type:"doorbell", camera:<friendly name>, event_id:<id>}.
        let camera = (dict["camera"] as? String) ?? "Doorbell"
        let eventID: Int?
        if let n = dict["event_id"] as? Int {
            eventID = n
        } else if let s = dict["event_id"] as? String {
            eventID = Int(s)
        } else {
            eventID = nil
        }

        reportIncomingCall(camera: camera, eventID: eventID, completion: completion)
    }
}

// MARK: - CXProviderDelegate

extension CallManager: CXProviderDelegate {

    func providerDidReset(_ provider: CXProvider) {
        activeCallUUID = nil
        activeCallCamera = nil
    }

    /// User answered: route the app to the doorbell live view, fulfill the
    /// action, then tear the CallKit call down so the app comes to the front.
    func provider(_ provider: CXProvider, perform action: CXAnswerCallAction) {
        if let camera = activeCallCamera {
            NotificationCenter.default.post(
                name: Self.openLiveCameraNotification,
                object: nil,
                userInfo: ["camera": camera]
            )
        }
        action.fulfill()
        // We don't hold a real audio call — the live view owns its own audio —
        // so end the CallKit call now that we've answered + routed.
        endActiveCall(reason: .answeredElsewhere)
    }

    /// Decline (before answer) or hang up (after) — tear down cleanly.
    func provider(_ provider: CXProvider, perform action: CXEndCallAction) {
        endActiveCall()          // already ended by the user via CallKit UI
        action.fulfill()
    }
}
