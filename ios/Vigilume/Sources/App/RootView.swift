import SwiftUI

/// Routes on auth state: restoring -> spinner, loggedOut -> LoginView,
/// loggedIn -> the main TabView.
struct RootView: View {
    @EnvironmentObject private var session: SessionModel

    var body: some View {
        ZStack {
            Theme.bg.ignoresSafeArea()
            switch session.phase {
            case .restoring:
                ProgressView()
                    .tint(Theme.accent)
            case .loggedOut:
                LoginView()
            case .loggedIn:
                MainTabView()
            }
        }
    }
}

enum AppTab: Hashable {
    case cameras
    case events
    case timeline
    case settings
}

struct MainTabView: View {
    @EnvironmentObject private var session: SessionModel
    @State private var selectedTab: AppTab = .cameras

    var body: some View {
        TabView(selection: $selectedTab) {
            CamerasView()
                .tabItem { Label("Cameras", systemImage: "video.fill") }
                .tag(AppTab.cameras)

            EventsView()
                .tabItem { Label("Events", systemImage: "bell.badge.fill") }
                .tag(AppTab.events)

            TimelineView()
                .tabItem { Label("Timeline", systemImage: "clock.fill") }
                .tag(AppTab.timeline)

            SettingsView()
                .tabItem { Label("Settings", systemImage: "gearshape.fill") }
                .tag(AppTab.settings)
        }
        // Deep link (vigilume://events/<id> or a tapped push) -> Events tab;
        // EventsView consumes session.pendingEventID to open the detail.
        .onChange(of: session.pendingEventID) { _, newValue in
            if newValue != nil { selectedTab = .events }
        }
        // Answered a doorbell call -> jump to Cameras; CamerasView opens the
        // live view for session.pendingLiveCameraName.
        .onChange(of: session.pendingLiveCameraName) { _, newValue in
            if newValue != nil { selectedTab = .cameras }
        }
        .onAppear {
            if session.pendingEventID != nil { selectedTab = .events }
            if session.pendingLiveCameraName != nil { selectedTab = .cameras }
        }
    }
}
