import SwiftUI

/// Settings › Recording: the iOS twin of the web's Settings → Recording
/// "Retention" card (frontend/src/pages/settings/RecordingTab.tsx) — how long
/// 24/7 footage, event clips and event snapshots stay on disk — so an admin can
/// set retention without opening the web app.
///
/// **Every write here is PATCH /api/settings — NEVER PUT.** PUT is a
/// full-document replace and every field carries a backend default, so any key
/// the body omits is RESET rather than left alone: a PUT missing
/// `notifications.apns.direct.p8` destroys the APNs signing key and silently
/// breaks push (verified empirically). PATCH deep-merges, so Save below sends
/// ONLY `{"recording": {...}}` and everything else is preserved untouched.
///
/// Admin-only: reached from `SettingsHomeView`'s `session.isAdmin` section, so
/// it lives inside that view's existing NavigationStack — no NavigationStack of
/// its own.
struct RecordingSettingsView: View {
    @EnvironmentObject private var session: SessionModel

    /// Last document loaded from the server — the baseline the drafts diff
    /// against to decide whether Save is enabled.
    @State private var doc: SettingsDocument?
    @State private var loading = true
    @State private var loadError: String?

    // Retention draft
    @State private var continuousDays = 7
    @State private var eventDays = 14
    @State private var snapshotDays = 14
    @State private var saving = false
    @State private var saveError: String?

    /// The backend validates each window with `Field(ge=0, le=365)`
    /// (RecordingSettings in backend/app/routers/settings.py) — the Steppers are
    /// bound to this range, so the UI can't produce a 422.
    private static let dayRange = 0 ... 365

    var body: some View {
        Group {
            if let loadError, doc == nil {
                ContentUnavailableView(
                    "Couldn't load settings",
                    systemImage: "externaldrive.badge.xmark",
                    description: Text(loadError)
                )
            } else {
                form
            }
        }
        .background(Theme.bg)
        .navigationTitle("Recording")
        .navigationBarTitleDisplayMode(.inline)
        .task { await load() }
        .refreshable { await load() }
    }

    private var form: some View {
        List {
            retentionSection
            saveSection
        }
        .scrollContentBackground(.hidden)
        .overlay {
            if loading && doc == nil {
                ProgressView().tint(Theme.accent)
            }
        }
        .disabled(doc == nil)
    }

    // MARK: - Retention

    /// Labels + hints mirror the web's three `dayInput`s verbatim.
    private var retentionSection: some View {
        Section {
            dayRow(
                "Continuous recording (days)",
                value: $continuousDays,
                hint: "24/7 footage kept on disk.",
                zeroHint: "Continuous footage is deleted as soon as the cleanup runs — effectively no 24/7 history."
            )
            dayRow(
                "Event clips (days)",
                value: $eventDays,
                hint: "Per-event recordings.",
                zeroHint: "Event clips are deleted as soon as the cleanup runs — events keep their snapshot, but no video."
            )
            dayRow(
                "Snapshots (days)",
                value: $snapshotDays,
                hint: "Event snapshot images. Snapshots are pruned together with their event record, at whichever is longer — this or Event clips — and never sooner than 1 day.",
                // 0 here is legal but NOT literally "keep nothing": the backend
                // prunes event rows (and their snapshots) at
                // max(event_days, snapshot_days, 1), so this field can only ever
                // extend retention beyond Event clips, never shorten it below.
                zeroHint: "Snapshots still follow the Event clips window (minimum 1 day) — this can lengthen snapshot retention past Event clips, not shorten it."
            )
        } header: {
            Text("Retention")
        } footer: {
            Text("How long recordings stay on disk before the cleanup removes them. Rule of thumb: continuous recording uses ≈ 10.8 GB per day for every 1 Mbps of combined camera bitrate (a typical 3-camera setup ≈ 135 GB/day, so 7 days ≈ 1 TB).")
        }
        .listRowBackground(Theme.surface)
    }

    /// One retention window: a 0…365 Stepper (the range clamps, so no 422) with
    /// the web's hint underneath, swapped for the meaning-of-zero copy at 0 so
    /// the field reads as "keep nothing" rather than broken/disabled.
    private func dayRow(
        _ label: String,
        value: Binding<Int>,
        hint: String,
        zeroHint: String
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Stepper(value: value, in: Self.dayRange) {
                HStack {
                    Text(label)
                        .foregroundStyle(Theme.textPrimary)
                    Spacer()
                    Text(dayValueText(value.wrappedValue))
                        .font(.subheadline.weight(.medium).monospacedDigit())
                        .foregroundStyle(value.wrappedValue == 0 ? Theme.warning : Theme.textPrimary)
                }
            }
            .tint(Theme.accent)
            .accessibilityValue(dayValueText(value.wrappedValue))

            Text(value.wrappedValue == 0 ? zeroHint : hint)
                .font(.caption)
                .foregroundStyle(value.wrappedValue == 0 ? Theme.warning : Theme.textSecondary)
        }
        .padding(.vertical, 2)
    }

    /// 0 is legal and means "keep nothing" — say so, rather than showing a bare
    /// "0 days" that reads like a disabled or unset field.
    private func dayValueText(_ days: Int) -> String {
        switch days {
        case 0: return "Keep nothing"
        case 1: return "1 day"
        default: return "\(days) days"
        }
    }

    // MARK: - Save

    private var dirty: Bool {
        guard let doc else { return false }
        return continuousDays != doc.recording.continuousDays
            || eventDays != doc.recording.eventDays
            || snapshotDays != doc.recording.snapshotDays
    }

    private var saveSection: some View {
        Section {
            Button {
                Task { await save() }
            } label: {
                HStack {
                    Spacer()
                    if saving {
                        ProgressView().tint(Theme.accent)
                    } else {
                        Text("Save retention")
                            .foregroundStyle(dirty ? Theme.accent : Theme.textSecondary)
                    }
                    Spacer()
                }
            }
            .disabled(saving || !dirty)

            if let saveError {
                Text(saveError)
                    .font(.footnote)
                    .foregroundStyle(Theme.danger)
            }
        } footer: {
            Text("Saves only the retention windows — nothing else on the server is touched. Retention is enforced by a background cleanup that runs on a timer (not the moment you save), so shortening a window can take up to an hour to take effect. Anything past its window is deleted permanently and cannot be recovered — files being written right now are always kept.")
        }
        .listRowBackground(Theme.surface)
    }

    private func save() async {
        guard let api = session.api, !saving else { return }
        saving = true
        defer { saving = false }
        saveError = nil
        // ONLY the recording subtree — the server deep-merges it and preserves
        // every other block (notifications/APNs key, detection, system, mqtt,
        // time_sync). Values come from Steppers bound to 0…365, so they're
        // already inside the backend's ge=0/le=365 validation.
        let patch = SettingsPatch(
            recording: .init(
                continuousDays: continuousDays,
                eventDays: eventDays,
                snapshotDays: snapshotDays
            )
        )
        do {
            apply(try await api.patchSettings(patch))
        } catch {
            session.handleAPIError(error)
            saveError = (error as? ApiError)?.message ?? error.localizedDescription
        }
    }

    // MARK: - Loading

    private func load() async {
        guard let api = session.api else { return }
        loading = doc == nil
        do {
            apply(try await api.settingsDocument())
            loadError = nil
        } catch {
            session.handleAPIError(error)
            loadError = (error as? ApiError)?.message ?? error.localizedDescription
        }
        loading = false
    }

    /// Adopt a server document (initial load or a PATCH response) as both the
    /// baseline and the draft — the response is the full updated document, so a
    /// save leaves the form showing exactly what the server now holds.
    private func apply(_ document: SettingsDocument) {
        doc = document
        continuousDays = document.recording.continuousDays
        eventDays = document.recording.eventDays
        snapshotDays = document.recording.snapshotDays
    }
}
