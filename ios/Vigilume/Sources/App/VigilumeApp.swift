import SwiftUI
import UserNotifications

@main
struct VigilumeApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    @StateObject private var serverStore: ServerStore
    @StateObject private var session: SessionModel

    init() {
        let store = ServerStore()
        _serverStore = StateObject(wrappedValue: store)
        _session = StateObject(wrappedValue: SessionModel(serverStore: store))
    }

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(serverStore)
                .environmentObject(session)
                .preferredColorScheme(.dark)   // security-console: dark only
                .tint(Theme.accent)
                .task {
                    // Warm the process-wide WebRTC factory on a background
                    // thread. Detached (never inherits this main-actor task's
                    // isolation) so RTCInitializeSSL + factory construction
                    // don't run on the main actor in the middle of the first
                    // live tile's layout. Idempotent and non-blocking: launch
                    // continues immediately.
                    Task.detached(priority: .utility) { WebRTCFactory.warm() }
                    await session.restore()
                    // Touch the push manager on every launch: re-registers
                    // with APNs (tokens can rotate) and re-POSTs to the
                    // active server when push is enabled for it
                    // (ios-design.md §3.2) — without waiting for the
                    // Settings tab to be opened.
                    if session.phase == .loggedIn {
                        PushManager.shared.syncToActiveServer()
                        // Same for the VoIP (CallKit doorbell) token: re-POST it
                        // to the active server so the phone can ring on a press.
                        CallManager.shared.syncToActiveServer()
                    }
                }
                .onOpenURL { url in
                    handleDeepLink(url)
                }
                .onReceive(NotificationCenter.default.publisher(
                    for: AppDelegate.openEventNotification
                )) { note in
                    guard let id = note.userInfo?["event_id"] as? Int else { return }
                    // E2E pushes name the server whose key decrypted them —
                    // switch to it first if it isn't the active one, so the
                    // event id resolves against the right NVR.
                    if let raw = note.userInfo?["server_id"] as? String,
                       let serverID = UUID(uuidString: raw),
                       serverID != serverStore.activeServerID,
                       serverStore.servers.contains(where: { $0.id == serverID }) {
                        Task {
                            await session.switchServer(to: serverID)
                            session.pendingEventID = id
                        }
                    } else {
                        session.pendingEventID = id
                    }
                }
                // Answered a CallKit doorbell call -> open that camera's live
                // view (CallManager posts the payload `camera` friendly name).
                .onReceive(NotificationCenter.default.publisher(
                    for: CallManager.openLiveCameraNotification
                )) { note in
                    guard let camera = note.userInfo?["camera"] as? String else { return }
                    session.pendingLiveCameraName = camera
                }
        }
    }

    /// vigilume://events/<id> (push payload deep_link; vigilume://event/<id>
    /// accepted as an alias) -> Events tab detail.
    private func handleDeepLink(_ url: URL) {
        guard url.scheme == "vigilume" else { return }
        let host = url.host ?? ""
        guard host == "events" || host == "event" else { return }
        let idPart = url.pathComponents.first { $0 != "/" }
        if let idPart, let id = Int(idPart) {
            session.pendingEventID = id
        }
    }
}

/// APNs registration scaffold. Push UI is gated on the backend route's
/// existence (POST /api/notifications/apns/register — 404 => unsupported).
final class AppDelegate: NSObject, UIApplicationDelegate, UNUserNotificationCenterDelegate {
    /// Posted with userInfo ["token": hexString] when APNs hands us a token.
    static let apnsTokenNotification = Notification.Name("VigilumeAPNsToken")
    /// Posted with userInfo ["event_id": Int] when a push is tapped.
    static let openEventNotification = Notification.Name("VigilumeOpenEvent")

    /// App-wide interface-orientation gate. Defaults to portrait; the
    /// fullscreen live view (SingleCameraView) widens this to allow landscape
    /// while it's on screen and resets it to `.portrait` on dismiss.
    static var orientationLock: UIInterfaceOrientationMask = .portrait

    func application(
        _ application: UIApplication,
        supportedInterfaceOrientationsFor window: UIWindow?
    ) -> UIInterfaceOrientationMask {
        AppDelegate.orientationLock
    }

    func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil
    ) -> Bool {
        let center = UNUserNotificationCenter.current()
        center.delegate = self
        // Our pushes carry no custom category, so they use the system default
        // category — which supports thread grouping and normal expand/collapse
        // of the per-camera stacks the NotificationService sets via
        // threadIdentifier. Registering an (empty) category set makes that
        // default behavior explicit and future-proofs adding actions later.
        center.setNotificationCategories([])
        // Register for PushKit VoIP pushes now, at launch — a VoIP push must be
        // able to wake the app to ring the phone (CallKit) even from cold, and
        // registration is independent of login. The token is only POSTed to the
        // backend once a server is signed in (CallManager.syncToActiveServer).
        CallManager.shared.registerForVoIPPushes()
        return true
    }

    func application(
        _ application: UIApplication,
        didRegisterForRemoteNotificationsWithDeviceToken deviceToken: Data
    ) {
        let hex = deviceToken.map { String(format: "%02x", $0) }.joined()
        NotificationCenter.default.post(
            name: Self.apnsTokenNotification, object: nil, userInfo: ["token": hex]
        )
    }

    func application(
        _ application: UIApplication,
        didFailToRegisterForRemoteNotificationsWithError error: Error
    ) {
        NSLog("APNs registration failed: \(error.localizedDescription)")
    }

    // Foreground pushes still show a banner.
    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        willPresent notification: UNNotification
    ) async -> UNNotificationPresentationOptions {
        [.banner, .sound]
    }

    // Tapped notification -> deep link to the event. E2E pushes carry
    // event_id (and server_id, set by the NotificationService extension from
    // whichever server's key decrypted the push) in userInfo; legacy
    // plaintext pushes carry event_id / deep_link directly.
    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        didReceive response: UNNotificationResponse
    ) async {
        let userInfo = response.notification.request.content.userInfo
        var eventID: Int?
        if let id = userInfo["event_id"] as? Int {
            eventID = id
        } else if let s = userInfo["event_id"] as? String, let id = Int(s) {
            eventID = id
        } else if let link = userInfo["deep_link"] as? String,
                  let url = URL(string: link),
                  let last = url.pathComponents.last, let id = Int(last) {
            eventID = id
        }
        if let eventID {
            var out: [String: Any] = ["event_id": eventID]
            if let serverID = userInfo["server_id"] as? String {
                out["server_id"] = serverID
            }
            NotificationCenter.default.post(
                name: Self.openEventNotification, object: nil, userInfo: out
            )
        }
    }
}
