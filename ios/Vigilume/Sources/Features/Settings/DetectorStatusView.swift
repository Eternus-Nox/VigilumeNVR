import SwiftUI

/// Settings › Detector (ADMIN ONLY): the detector self-test and per-camera
/// ingest health — the iOS twin of the web's detector status panel. It reads
/// GET /api/system/detector (`require_admin`) into `DetectorStatus`.
///
/// Read-mostly, mirroring `SuppressionsView`/`UsersView`: load in `.task`, a
/// Refresh button in the toolbar plus pull-to-refresh, and load failures surface
/// via `session.handleAPIError` + an `errorMessage` rather than a dead spinner.
/// The whole endpoint is admin-only, so we also gate the load on
/// `session.isAdmin` — a viewer that somehow lands here gets a message, not a
/// 403 loop. Reached from `SettingsHomeView`'s `session.isAdmin` section, so it
/// lives inside that view's existing NavigationStack — no NavigationStack of its
/// own.
struct DetectorStatusView: View {
    @EnvironmentObject private var session: SessionModel

    @State private var status: DetectorStatus?
    @State private var loading = true
    @State private var errorMessage: String?

    var body: some View {
        Group {
            if !session.isAdmin {
                ContentUnavailableView(
                    "Admins only",
                    systemImage: "lock.fill",
                    description: Text("Detector status is restricted to admin accounts.")
                )
            } else if let errorMessage, status == nil {
                ContentUnavailableView(
                    "Couldn't load detector",
                    systemImage: "cpu",
                    description: Text(errorMessage)
                )
            } else if let status {
                list(status)
            } else {
                // Initial load — the overlay below carries the spinner.
                Color.clear
            }
        }
        .background(Theme.bg)
        .navigationTitle("Detector")
        .navigationBarTitleDisplayMode(.inline)
        .overlay {
            if loading && status == nil {
                ProgressView().tint(Theme.accent)
            }
        }
        .toolbar {
            if session.isAdmin {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        Task { await load() }
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                    .accessibilityLabel("Refresh")
                    .disabled(loading)
                }
            }
        }
        .task { await load() }
        .refreshable { await load() }
    }

    // MARK: List

    private func list(_ status: DetectorStatus) -> some View {
        List {
            detectorSection(status)
            if isDownloading(status) {
                downloadSection(status)
            }
            camerasSection(status)
        }
        .scrollContentBackground(.hidden)
    }

    // MARK: Detector summary

    private func detectorSection(_ status: DetectorStatus) -> some View {
        Section {
            // Ready — the headline signal, so it leads with a status dot.
            HStack {
                Circle()
                    .fill(status.ready ? Theme.success : Theme.danger)
                    .frame(width: 9, height: 9)
                    .accessibilityHidden(true)
                Text(status.ready ? "Ready" : "Not ready")
                    .foregroundStyle(status.ready ? Theme.success : Theme.danger)
                Spacer(minLength: 0)
            }
            .font(.subheadline.weight(.medium))
            .accessibilityElement(children: .combine)
            .accessibilityLabel("Detector \(status.ready ? "ready" : "not ready")")

            infoRow("Device", friendlyDevice(status.device))
            infoRow("Model", status.model)

            if let shaOk = status.modelShaOk {
                infoRow(
                    "Model checksum",
                    shaOk ? "OK" : "Mismatch",
                    valueColor: shaOk ? Theme.success : Theme.danger
                )
            }
            if let ms = status.lastInferenceMs {
                infoRow("Last inference", String(format: "%.1f ms", ms))
            }
            if let failures = status.consecutiveFailures {
                infoRow(
                    "Consecutive failures",
                    "\(failures)",
                    valueColor: failures > 0 ? Theme.warning : Theme.textPrimary
                )
            }
            if status.needsReinit == true {
                infoRow("Needs re-init", "Yes", valueColor: Theme.warning)
            }
            if let ageS = status.lastReinitAgeS {
                infoRow("Last re-init", ageText(ageS))
            }
        } header: {
            Text("Detector")
        }
        .listRowBackground(Theme.surface)
    }

    // MARK: Model download

    /// Only shown mid-download — `modelState` reports "downloading" while the
    /// backend pulls a model, with `modelProgressPct` as its 0–100 progress.
    private func downloadSection(_ status: DetectorStatus) -> some View {
        Section {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text((status.modelState ?? "downloading").capitalized)
                        .font(.subheadline)
                        .foregroundStyle(Theme.textPrimary)
                    Spacer(minLength: 12)
                    if let pct = status.modelProgressPct {
                        Text("\(pct)%")
                            .font(.subheadline)
                            .foregroundStyle(Theme.textSecondary)
                    }
                }
                ProgressView(value: Double(status.modelProgressPct ?? 0), total: 100)
                    .tint(Theme.accent)
            }
            .padding(.vertical, 2)
        } header: {
            Text("Model download")
        }
        .listRowBackground(Theme.surface)
    }

    // MARK: Cameras

    private func camerasSection(_ status: DetectorStatus) -> some View {
        Section {
            if status.perCamera.isEmpty {
                Text("No cameras reporting.")
                    .font(.subheadline)
                    .foregroundStyle(Theme.textSecondary)
            } else {
                ForEach(status.perCamera) { camera in
                    cameraRow(camera)
                }
            }
        } header: {
            Text("Cameras")
        } footer: {
            Text("Per-camera frame ingest into the detector. “Stalled” means frames stopped arriving; respawns count how often the capture worker has restarted.")
        }
        .listRowBackground(Theme.surface)
    }

    private func cameraRow(_ camera: DetectorStatus.Camera) -> some View {
        HStack(spacing: 12) {
            Circle()
                .fill(dotColor(for: camera))
                .frame(width: 9, height: 9)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 3) {
                Text(camera.name)
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(Theme.textPrimary)
                Text(cameraSubtitle(camera))
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
            }
            Spacer(minLength: 8)
            HStack(spacing: 6) {
                if camera.aiActive == true {
                    badge("AI", tint: Theme.accent)
                }
                if camera.stalled == true {
                    badge("Stalled", tint: Theme.warning)
                }
            }
        }
        .padding(.vertical, 4)
        .accessibilityElement(children: .combine)
    }

    /// "12.0 fps · 0.3s ago · 3 respawns" — pieces drop out when the backend
    /// omits them rather than showing a bare "—" for everything.
    private func cameraSubtitle(_ camera: DetectorStatus.Camera) -> String {
        var parts: [String] = []
        if let fps = camera.fps {
            parts.append(String(format: "%.1f fps", fps))
        } else {
            parts.append("— fps")
        }
        if let age = camera.lastFrameAgeS {
            parts.append(String(format: "%.1fs ago", age))
        } else {
            parts.append("—")
        }
        if let respawns = camera.respawns, respawns > 0 {
            parts.append("\(respawns) respawn\(respawns == 1 ? "" : "s")")
        }
        return parts.joined(separator: " · ")
    }

    // MARK: Helpers

    /// A key/value line matching the theme (label secondary, value primary).
    private func infoRow(
        _ label: String,
        _ value: String,
        valueColor: Color = Theme.textPrimary
    ) -> some View {
        HStack {
            Text(label)
                .foregroundStyle(Theme.textSecondary)
            Spacer(minLength: 12)
            Text(value)
                .foregroundStyle(valueColor)
                .multilineTextAlignment(.trailing)
        }
        .font(.subheadline)
    }

    private func badge(_ text: String, tint: Color) -> some View {
        Text(text)
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(tint.opacity(0.18))
            .foregroundStyle(tint)
            .clipShape(Capsule())
    }

    private func dotColor(for camera: DetectorStatus.Camera) -> Color {
        if camera.stalled == true { return Theme.warning }
        if camera.ingestOk == true { return Theme.success }
        if camera.ingestOk == false { return Theme.danger }
        return Theme.textSecondary
    }

    /// Friendly device names, matching the web's describeDetector — the raw
    /// backend values are "cuda" / "cpu" / "edgetpu".
    private func friendlyDevice(_ device: String) -> String {
        switch device {
        case "edgetpu": return "Coral Edge TPU"
        case "cuda": return "GPU"
        case "cpu": return "CPU"
        default: return device.uppercased()
        }
    }

    /// Compact age: "12s" / "4m" / "1.2h".
    private func ageText(_ seconds: Double) -> String {
        if seconds < 90 { return String(format: "%.0fs ago", seconds) }
        if seconds < 5400 { return String(format: "%.0fm ago", seconds / 60) }
        return String(format: "%.1fh ago", seconds / 3600)
    }

    private func isDownloading(_ status: DetectorStatus) -> Bool {
        status.modelState?.lowercased() == "downloading"
    }

    // MARK: Loading

    private func load() async {
        guard session.isAdmin, let api = session.api else { return }
        loading = status == nil
        do {
            status = try await api.detector()
            errorMessage = nil
        } catch {
            session.handleAPIError(error)
            errorMessage = (error as? ApiError)?.message ?? error.localizedDescription
        }
        loading = false
    }
}
