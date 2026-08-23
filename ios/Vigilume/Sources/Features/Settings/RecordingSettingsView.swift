import SwiftUI

/// Settings › Recording: the iOS twin of the web's Settings → Recording tab
/// (frontend/src/pages/settings/RecordingTab.tsx) — retention windows, the
/// space-based storage limits, event clip padding and event grouping — so an
/// admin can tune recording without opening the web app.
///
/// Two settings subtrees are edited here: `recording` for everything except
/// event grouping, which is `detection.absence_timeout_s`. They are patched
/// independently (see `save()`), so editing one never writes the other.
///
/// **Every write here is PATCH /api/settings — NEVER PUT.** PUT is a
/// full-document replace and every field carries a backend default, so any key
/// the body omits is RESET rather than left alone: a PUT missing
/// `notifications.apns.direct.p8` destroys the APNs signing key and silently
/// breaks push (verified empirically). PATCH deep-merges, so Save below sends
/// only the subtrees edited here — and only the ones that actually changed —
/// leaving everything else preserved untouched.
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
    // Storage-limit draft (GiB)
    @State private var maxStorageGb = 0
    @State private var minFreeGb = 5
    // Clip-timing draft (seconds)
    @State private var clipPreS = 5
    @State private var clipPostS = 5
    @State private var clipDelayS = 20
    // Event grouping (detection subtree, edited here because it decides where a
    // clip ends — the web groups it with the clip settings for the same reason)
    @State private var absenceTimeoutS = 5
    @State private var saving = false
    @State private var saveError: String?

    /// The backend validates each window with `Field(ge=0, le=365)`
    /// (RecordingSettings in backend/app/routers/settings.py) — the Steppers are
    /// bound to this range, so the UI can't produce a 422.
    private static let dayRange = 0 ... 365
    /// Matching backend bounds for the rest. Steppers are bound to these, so
    /// like `dayRange` above the UI cannot produce a 422.
    private static let clipPreRange = 0 ... 120
    private static let clipDelayRange = 10 ... 300
    private static let absenceRange = 1 ... 300
    private static let minFreeRange = 1 ... 10_000

    /// Reachable post-roll at the currently drafted delay. The ceiling MOVES
    /// with the delay, so the Stepper's range is derived rather than fixed —
    /// a field that silently refuses to go past 10 with no explanation is a
    /// worse way to learn the rule than watching it rise as the delay does.
    private var maxClipPostS: Int { SettingsDocument.maxClipPostS(clipDelayS) }

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
            storageLimitsSection
            clipPaddingSection
            eventGroupingSection
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

    // MARK: - Storage limits

    /// Mirrors the web's "Storage limits" card. Space-based rotation runs on its
    /// own minute timer, independently of the day windows above.
    private var storageLimitsSection: some View {
        Section {
            valueRow(
                "Maximum recording storage",
                value: $maxStorageGb,
                range: 0 ... 1_000_000,
                step: 100,
                display: { $0 == 0 ? "No cap" : "\($0) GB" },
                hint: maxStorageGb == 0
                    ? "No cap — recordings grow until the free-space floor below stops them."
                    : "Recordings are held under this. Set it when the disk is shared with other data."
            )
            valueRow(
                "Keep free space",
                value: $minFreeGb,
                range: Self.minFreeRange,
                step: 1,
                display: { "\($0) GB" },
                hint: "Always leave at least this much free, whatever the cap."
            )
        } header: {
            Text("Storage limits")
        } footer: {
            Text("When space runs out the oldest 24/7 footage is deleted to make room for the newest, checked every minute. This applies on top of the day windows above — whichever frees a recording first wins. Event clips are never deleted for space; they expire only by their own retention.")
        }
        .listRowBackground(Theme.surface)
    }

    // MARK: - Clip padding

    private var clipPaddingSection: some View {
        Section {
            valueRow(
                "Lead-in before event",
                value: $clipPreS,
                range: Self.clipPreRange,
                step: 1,
                display: secondsText,
                hint: clipPreS == 0
                    ? "The clip starts exactly at detection — by which point a subject may already be well into frame."
                    : "Try 15 s if clips start too late."
            )
            valueRow(
                "Run-on after event",
                value: $clipPostS,
                range: 0 ... max(0, maxClipPostS),
                step: 1,
                display: secondsText,
                hint: "Limited to \(maxClipPostS) s by the cut delay below — later footage is not on disk yet when the clip is assembled."
            )
            valueRow(
                "Cut the clip after",
                value: $clipDelayS,
                range: Self.clipDelayRange,
                step: 5,
                display: secondsText,
                hint: "Only the clip waits — the event, its snapshot and its notification arrive immediately. Raise it to allow more run-on.",
                // Lowering the delay lowers what run-on can reach. Without this
                // the pair goes out of range and the save 422s naming
                // clip_post_s, when the mistake was made in clip_delay_s.
                onChange: { clipPostS = min(clipPostS, SettingsDocument.maxClipPostS($0)) }
            )
        } header: {
            Text("Clip padding")
        } footer: {
            Text("Extra footage kept either side of an event. Both are measured from when the object was DETECTED, which is later than when it entered frame — the tracker needs a few frames on something big enough to recognise, and a subject approaching from a distance can be visible for seconds before that. The footage is copied from 24/7 recording already on disk, so wider padding costs a little clip storage and no extra CPU.")
        }
        .listRowBackground(Theme.surface)
    }

    // MARK: - Event grouping

    /// `detection.absence_timeout_s`. A DIFFERENT settings subtree from
    /// everything above, patched separately in `save()` — grouped here because
    /// it decides when an event (and so its clip) ends.
    private var eventGroupingSection: some View {
        Section {
            valueRow(
                "End the event after",
                value: $absenceTimeoutS,
                range: Self.absenceRange,
                step: 1,
                display: { "\($0) s unseen" },
                hint: "Clips are only cut once the event ends, so this also delays when a clip appears."
            )
        } header: {
            Text("Event grouping")
        } footer: {
            Text("How long an object may go unseen before its event is closed. This adds no footage — an event still ends at the last frame the object was actually seen — but it decides whether a subject that pauses, turns away, or slips behind cover becomes one event or several. Raise it for scenes with obstructions; lower it if separate visits are being merged.")
        }
        .listRowBackground(Theme.surface)
    }

    private func secondsText(_ s: Int) -> String { s == 1 ? "1 second" : "\(s) seconds" }

    /// The generalised twin of `dayRow`: a clamped Stepper with a caption, for
    /// the fields whose range and formatting are not "0…365 days".
    ///
    /// `onChange` exists for the one cross-field rule here — the cut delay
    /// bounding run-on. Steppers clamp their OWN value to `range`, but a value
    /// already drafted stays put when another field narrows that range, so the
    /// dependent field has to be pulled down explicitly.
    private func valueRow(
        _ label: String,
        value: Binding<Int>,
        range: ClosedRange<Int>,
        step: Int,
        display: @escaping (Int) -> String,
        hint: String,
        onChange: ((Int) -> Void)? = nil
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Stepper(value: value, in: range, step: step) {
                HStack {
                    Text(label)
                        .foregroundStyle(Theme.textPrimary)
                    Spacer()
                    Text(display(value.wrappedValue))
                        .font(.subheadline.weight(.medium).monospacedDigit())
                        .foregroundStyle(Theme.textPrimary)
                }
            }
            .tint(Theme.accent)
            .accessibilityValue(display(value.wrappedValue))
            .onChange(of: value.wrappedValue) { _, new in onChange?(new) }

            Text(hint)
                .font(.caption)
                .foregroundStyle(Theme.textSecondary)
        }
        .padding(.vertical, 2)
    }

    // MARK: - Save

    private var recordingDirty: Bool {
        guard let doc else { return false }
        return continuousDays != doc.recording.continuousDays
            || eventDays != doc.recording.eventDays
            || snapshotDays != doc.recording.snapshotDays
            || maxStorageGb != doc.recording.maxStorageGb
            || minFreeGb != doc.recording.minFreeGb
            || clipPreS != doc.recording.clipPreS
            || clipPostS != doc.recording.clipPostS
            || clipDelayS != doc.recording.clipDelayS
    }

    /// Tracked separately from `recordingDirty` because it lives in the
    /// `detection` subtree: `save()` sends that patch ONLY when this is true,
    /// so touching a retention window never writes to detection at all.
    private var detectionDirty: Bool {
        guard let doc else { return false }
        return absenceTimeoutS != doc.detection.absenceTimeoutS
    }

    private var dirty: Bool { recordingDirty || detectionDirty }

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
                        Text("Save")
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
            Text("Saves only what you changed on this screen — nothing else on the server is touched. Day-based retention is enforced by a cleanup on an hourly timer, so shortening a window can take up to an hour to take effect; the storage limits are checked every minute. Clip padding and event grouping apply to the next event, and change nothing already recorded. Anything deleted is gone permanently — files being written right now are always kept.")
        }
        .listRowBackground(Theme.surface)
    }

    private func save() async {
        guard let api = session.api, !saving else { return }
        saving = true
        defer { saving = false }
        saveError = nil
        // Only the subtrees this screen actually edits, and only the ones that
        // CHANGED — the server deep-merges, so every other block
        // (notifications/APNs key, system, mqtt, time_sync) is preserved, and
        // an untouched `detection` is not written at all. That last part
        // matters: the detector model is activated out-of-band, so a patch
        // naming `detection` when nothing here changed is a chance to race it.
        //
        // Every value comes from a Stepper bound to the backend's own range, so
        // none of this can 422 — including clipPostS, whose Stepper range is
        // derived from the drafted clipDelayS.
        let patch = SettingsPatch(
            recording: recordingDirty
                ? .init(
                    continuousDays: continuousDays,
                    eventDays: eventDays,
                    snapshotDays: snapshotDays,
                    maxStorageGb: maxStorageGb,
                    minFreeGb: minFreeGb,
                    clipPreS: clipPreS,
                    clipPostS: clipPostS,
                    clipDelayS: clipDelayS
                )
                : nil,
            detection: detectionDirty ? .init(absenceTimeoutS: absenceTimeoutS) : nil
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
        maxStorageGb = document.recording.maxStorageGb
        minFreeGb = document.recording.minFreeGb
        clipPreS = document.recording.clipPreS
        clipPostS = document.recording.clipPostS
        clipDelayS = document.recording.clipDelayS
        absenceTimeoutS = document.detection.absenceTimeoutS
    }
}
