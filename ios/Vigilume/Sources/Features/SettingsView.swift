import SwiftUI

/// Settings tab: account, push notifications (APNs registration, gated on the
/// backend route's existence), server list (add/switch/remove), About, and an
/// admin link-out to the web app. Implementation lives in
/// Features/Settings/SettingsHomeView.swift; PushManager.shared is resolved
/// here (body is MainActor) and injected so the notification state observes.
struct SettingsView: View {
    var body: some View {
        SettingsHomeView(push: PushManager.shared)
    }
}
