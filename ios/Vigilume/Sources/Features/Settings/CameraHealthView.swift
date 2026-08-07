import SwiftUI

/// Settings › Camera health: the iOS twin of the web's `CameraHealthCard`
/// (frontend/src/components/CameraHealthCard.tsx). A per-camera board of RTSP
/// stream-port reachability over a chosen window — a "now" status dot, uptime %,
/// outage count and total downtime for each camera.
///
/// **"Uptime" here is CONNECTIVITY, not a guarantee footage was recorded** — it's
/// only what the reachability prober measures, and the footer says so rather than
/// implying video was written to disk.
///
/// The endpoint (`GET /api/system/camera-health`) is `require_auth`, so this view
/// is viewer-visible, not admin-only. It's pushed from within `SettingsHomeView`'s
/// existing NavigationStack, so it carries no NavigationStack of its own.
///
/// An older backend won't have the endpoint at all; a 404/405 is treated as
/// "not available on this server" (a graceful message) rather than an error —
/// mirroring how the web card hides itself when the route is absent.
struct CameraHealthView: View {
    @EnvironmentObject private var session: SessionModel

    /// Selected window in hours: 24 (24h) / 168 (7d) / 720 (30d).
    @State private var hours = 24
    @State private var report: CameraHealthReport?
    @State private var loading = true
    /// Set when a fetch fails for any reason OTHER than the endpoint being
    /// absent — shown in place of the board when there's nothing to display.
    @State private var errorMessage: String?
    /// The endpoint 404/405'd — an older backend that never shipped it. Shown as
    /// a calm "not available" state, not an error.
    @State private var unavailable = false

    /// The segmented window choices, in web order (24h / 7d / 30d).
    private struct WindowOption: Identifiable {
        let hours: Int
        let label: String
        var id: Int { hours }
    }
    private static let windows: [WindowOption] = [
        .init(hours: 24, label: "24h"),
        .init(hours: 24 * 7, label: "7d"),
        .init(hours: 24 * 30, label: "30d"),
    ]

    var body: some View {
        Group {
            if unavailable {
                ContentUnavailableView(
                    "Camera health unavailable",
                    systemImage: "wifi.exclamationmark",
                    description: Text("This server doesn't report camera reachability yet. Update the NVR to enable it.")
                )
            } else if let errorMessage, report == nil {
                ContentUnavailableView(
                    "Couldn't load camera health",
                    systemImage: "wifi.exclamationmark",
                    description: Text(errorMessage)
                )
            } else {
                board
            }
        }
        .background(Theme.bg)
        .navigationTitle("Camera health")
        .navigationBarTitleDisplayMode(.inline)
        .onChange(of: hours) { _, _ in Task { await load() } }
        .task { await load() }
        .refreshable { await load() }
    }

    // MARK: - Board

    private var board: some View {
        List {
            Section {
                Picker("Window", selection: $hours) {
                    ForEach(Self.windows) { option in
                        Text(option.label).tag(option.hours)
                    }
                }
                .pickerStyle(.segmented)
            }
            .listRowBackground(Theme.surface)

            Section {
                if let report {
                    if report.cameras.isEmpty {
                        Text("No cameras")
                            .font(.subheadline)
                            .foregroundStyle(Theme.textSecondary)
                            .listRowBackground(Theme.surface)
                    } else {
                        ForEach(report.cameras) { camera in
                            row(camera)
                                .listRowBackground(Theme.surface)
                        }
                    }
                }
            } footer: {
                Text("Uptime is reachability of each camera's stream port over the selected window — connectivity, not a guarantee that footage was recorded.")
            }
        }
        .scrollContentBackground(.hidden)
        .overlay {
            if loading && report == nil {
                ProgressView().tint(Theme.accent)
            }
        }
    }

    // MARK: - Row

    /// One camera: a live "now" dot + friendly name on the leading edge, its
    /// outage count and total downtime beneath, and the window's uptime % on the
    /// trailing edge.
    private func row(_ camera: CameraHealthReport.Row) -> some View {
        HStack(spacing: 12) {
            Circle()
                .fill(dotColor(camera.online))
                .frame(width: 9, height: 9)
                .accessibilityLabel(statusLabel(camera.online))
            VStack(alignment: .leading, spacing: 3) {
                Text(displayName(camera.camera))
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(Theme.textPrimary)
                Text(subtitle(camera))
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
            }
            Spacer(minLength: 0)
            VStack(alignment: .trailing, spacing: 2) {
                Text(camera.uptimePct.map { "\(pctText($0))%" } ?? "—")
                    .font(.subheadline.weight(.semibold).monospacedDigit())
                    .foregroundStyle(Theme.textPrimary)
                Text("uptime")
                    .font(.caption2)
                    .foregroundStyle(Theme.textSecondary)
            }
        }
        .padding(.vertical, 4)
    }

    /// "N outages · Nm down" — the secondary metrics line under the camera name.
    private func subtitle(_ camera: CameraHealthReport.Row) -> String {
        let outages = camera.downCount == 1 ? "1 outage" : "\(camera.downCount) outages"
        let downtime = camera.downSeconds > 0 ? "\(fmtDuration(camera.downSeconds)) down" : "no downtime"
        return "\(outages) · \(downtime)"
    }

    // MARK: - Formatting

    /// Status dot colour: green online, red offline, grey unknown (`online` nil).
    private func dotColor(_ online: Bool?) -> Color {
        switch online {
        case .some(true): return Theme.success
        case .some(false): return Theme.danger
        case .none: return Theme.textSecondary
        }
    }

    private func statusLabel(_ online: Bool?) -> String {
        switch online {
        case .some(true): return "Online"
        case .some(false): return "Offline"
        case .none: return "Unknown"
        }
    }

    /// Uptime percentage without noisy trailing zeros: "99" not "99.0", but
    /// "99.5" keeps its decimal.
    private func pctText(_ value: Double) -> String {
        value == value.rounded() ? String(Int(value)) : String(format: "%.1f", value)
    }

    /// Human downtime, mirroring the web card's `fmtDuration`:
    /// <60s → "Ns", <1h → "Nm", <1d → "N.Nh", else "N.Nd".
    private func fmtDuration(_ seconds: Double) -> String {
        if seconds < 60 { return "\(Int(seconds.rounded()))s" }
        if seconds < 3600 { return "\(Int((seconds / 60).rounded()))m" }
        if seconds < 86400 { return String(format: "%.1fh", seconds / 3600) }
        return String(format: "%.1fd", seconds / 86400)
    }

    /// Friendly name from the camera slug (front_yard → Front Yard) — the report
    /// carries only the slug, and this matches the web's `titleCase(cam.name)`
    /// fallback used across the app.
    private func displayName(_ camera: String) -> String {
        camera.replacingOccurrences(of: "_", with: " ").capitalized
    }

    // MARK: - Loading

    private func load() async {
        guard let api = session.api else { return }
        loading = report == nil
        do {
            report = try await api.cameraHealth(hours: hours)
            errorMessage = nil
            unavailable = false
        } catch {
            session.handleAPIError(error)
            // A missing route (older backend) is "not available", not an error.
            if let apiError = error as? ApiError, apiError.status == 404 || apiError.status == 405 {
                unavailable = true
            } else {
                errorMessage = (error as? ApiError)?.message ?? error.localizedDescription
            }
        }
        loading = false
    }
}
