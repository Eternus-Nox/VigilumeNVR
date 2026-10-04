/// Camera MAIN-stream quality — resolution, codec, keyframe interval, bitrate
/// (backend amcrest/encode.py + stream_profiles.py). Mirrors the web
/// MainStreamCard:
///
///   • `MainStreamAllCamerasView` (Settings → Cameras → Video quality): the
///     profile EVERY camera follows, applied to all of them at once, with what
///     each camera actually did.
///   • `CameraMainStreamView` (a camera's settings → Video quality): that
///     camera's stream as it is now, read live, and either "same as all
///     cameras" or its own profile.
///
/// Both apply IMMEDIATELY and show the camera's read-back — a camera answering
/// OK and keeping the old value is a real thing these cameras do. The backend
/// re-applies on reconnect and every 30 min, so a reset camera drifts back.
///
/// ADMIN-ONLY: every route here is `require_admin`, and both screens are only
/// reachable from admin-gated rows.
import SwiftUI

// MARK: - Models

struct MainStreamProfile: Codable, Equatable {
    var resolution: String = "keep"
    var codec: String = "keep"
    var keyframeS: Double?
    var bitrateKbps: Int?

    static let keep = MainStreamProfile()
    /// Lighter to decode (~4-5x less than 4K H.265), plays in live view on
    /// every device, and a short wait for the first frame.
    static let recommended = MainStreamProfile(
        resolution: "1080p", codec: "h264", keyframeS: 1, bitrateKbps: 4096
    )

    var summary: String {
        var parts: [String] = []
        if resolution != "keep" {
            parts.append(MainStreamProfile.resolutionChoices.first { $0.value == resolution }?.label
                         ?? resolution.replacingOccurrences(of: "x", with: "×"))
        }
        if codec != "keep" { parts.append(codec == "h264" ? "H.264" : "H.265") }
        if let keyframeS { parts.append("keyframe every \(MainStreamProfile.seconds(keyframeS))") }
        if let bitrateKbps { parts.append("\(bitrateKbps) kbps") }
        return parts.isEmpty ? "Unchanged" : parts.joined(separator: " · ")
    }

    struct Choice: Hashable {
        let value: String
        let label: String
    }

    static let resolutionChoices: [Choice] = [
        Choice(value: "keep", label: "Leave as is"),
        Choice(value: "720p", label: "Up to 720p"),
        Choice(value: "1080p", label: "Up to 1080p"),
        Choice(value: "1440p", label: "Up to 1440p"),
        Choice(value: "4k", label: "Up to 4K"),
        Choice(value: "max", label: "Camera maximum"),
    ]
    static let codecChoices: [Choice] = [
        Choice(value: "keep", label: "Leave as is"),
        Choice(value: "h264", label: "H.264"),
        Choice(value: "h265", label: "H.265"),
    ]

    static func seconds(_ s: Double) -> String {
        s == s.rounded() ? "\(Int(s)) s" : String(format: "%.1f s", s)
    }
}

struct MainStreamResult: Decodable, Identifiable {
    let camera: String
    let ok: Bool
    let skipped: Bool?
    let error: String?
    let changed: [String]?
    let rejected: [String]?
    let notApplied: [String]?
    let notes: [String]?

    var id: String { camera }

    /// The regular stream's changes only ("main …", not the event-time
    /// "main #1 …" copies kept in step with it).
    var summary: String {
        if let error { return error }
        if skipped == true { return "Nothing to change" }
        let main = (changed ?? [])
            .filter { $0.hasPrefix("main ") && !$0.hasPrefix("main #") }
            .map { String($0.dropFirst(5)).replacingOccurrences(of: " -> ", with: " → ") }
        let problems = ((rejected ?? []) + (notApplied ?? []))
            .map { $0.replacingOccurrences(of: " -> ", with: " → ") }
        if main.isEmpty && problems.isEmpty { return "Already set" }
        var lines: [String] = []
        if !main.isEmpty { lines.append("Changed: " + main.joined(separator: "; ")) }
        if !problems.isEmpty { lines.append("Not changed: " + problems.joined(separator: "; ")) }
        return lines.joined(separator: "\n")
    }
}

struct MainStreamNow: Decodable {
    let width: Int?
    let height: Int?
    let codecRaw: String?
    let fps: Int?
    let keyframeS: Double?
    let bitrateKbps: Int?
    let bitrateControl: String?

    var summary: String {
        let size = (width != nil && height != nil) ? "\(width!)×\(height!)" : "?"
        let key = keyframeS.map { MainStreamProfile.seconds($0) } ?? "?"
        let rate = bitrateKbps.map { "\($0) kbps" } ?? "? kbps"
        return "\(size) · \(codecRaw ?? "?") · \(fps.map { "\($0)" } ?? "?") fps · "
            + "keyframe every \(key) · \(rate)\(bitrateControl.map { " \($0)" } ?? "")"
    }
}

struct CameraMainStreamRow: Decodable {
    let camera: String
    let inherited: Bool
    let own: MainStreamProfile?
    let effective: MainStreamProfile
    let last: MainStreamResult?
}

struct MainStreamOverview: Decodable {
    let profile: MainStreamProfile
    let cameras: [CameraMainStreamRow]
}

struct CameraMainStreamDetail: Decodable {
    struct Size: Decodable, Hashable {
        let label: String
        let width: Int
        let height: Int
    }

    /// The live-view substream (ExtraFormat[0]).
    struct Sub: Decodable {
        let codec: String?
        let codecRaw: String?
        let width: Int?
        let height: Int?

        var isH264: Bool { codec == "h264" }
        var summary: String {
            let size = (width != nil && height != nil) ? " \(width!)×\(height!)" : ""
            return "Live-view substream: \(codecRaw ?? "?")\(size)"
        }
    }

    struct Live: Decodable {
        let current: MainStreamNow?
        let resolutions: [Size]
        /// Absent on an older backend.
        let substream: Sub?
    }

    let camera: String
    let inherited: Bool
    let own: MainStreamProfile?
    let effective: MainStreamProfile
    let global: MainStreamProfile?
    let live: Live?
    let error: String?
    let result: MainStreamResult?
}

struct MainStreamResults: Decodable {
    let results: [MainStreamResult]
}

// MARK: - API

extension APIClient {
    func mainStreamOverview() async throws -> MainStreamOverview {
        let data = try await controlsSend("GET", "api/cameras/main-stream")
        return try ControlsJSON.decoder.decode(MainStreamOverview.self, from: data)
    }

    /// Store the all-cameras profile and apply it now (waits for the cameras).
    func setMainStreamForAll(_ profile: MainStreamProfile, resetCameras: Bool) async throws -> MainStreamResults {
        struct Body: Encodable {
            let profile: MainStreamProfile
            let resetCameras: Bool
        }
        let body = try ControlsJSON.encoder.encode(Body(profile: profile, resetCameras: resetCameras))
        let data = try await controlsSend("PUT", "api/cameras/main-stream", body: body)
        return try ControlsJSON.decoder.decode(MainStreamResults.self, from: data)
    }

    func reapplyMainStreams() async throws -> MainStreamResults {
        let data = try await controlsSend("POST", "api/cameras/main-stream/apply")
        return try ControlsJSON.decoder.decode(MainStreamResults.self, from: data)
    }

    /// One camera's profile plus its main stream as it is now (read live).
    func cameraMainStream(_ name: String) async throws -> CameraMainStreamDetail {
        let data = try await controlsSend("GET", "api/cameras/\(name)/main-stream")
        return try ControlsJSON.decoder.decode(CameraMainStreamDetail.self, from: data)
    }

    /// Pin this camera's own profile (applied now), or nil to follow all cameras.
    func setCameraMainStream(_ name: String, profile: MainStreamProfile?) async throws -> CameraMainStreamDetail {
        struct Body: Encodable {
            let profile: MainStreamProfile?
        }
        let body = try ControlsJSON.encoder.encode(Body(profile: profile))
        let data = try await controlsSend("PUT", "api/cameras/\(name)/main-stream", body: body)
        return try ControlsJSON.decoder.decode(CameraMainStreamDetail.self, from: data)
    }
}

// MARK: - Shared form

/// The four settings, as Form rows. `sizes` adds one camera's exact sizes.
struct MainStreamProfileFields: View {
    @Binding var profile: MainStreamProfile
    var sizes: [CameraMainStreamDetail.Size] = []

    @State private var bitrateText = ""

    var body: some View {
        Picker("Resolution", selection: $profile.resolution) {
            ForEach(MainStreamProfile.resolutionChoices, id: \.value) { choice in
                Text(choice.label).tag(choice.value)
            }
            ForEach(sizes, id: \.self) { size in
                Text("\(size.width)×\(size.height)").tag("\(size.width)x\(size.height)")
            }
        }
        Picker("Codec", selection: $profile.codec) {
            ForEach(MainStreamProfile.codecChoices, id: \.value) { choice in
                Text(choice.label).tag(choice.value)
            }
        }
        Picker("Keyframe every", selection: $profile.keyframeS) {
            Text("Leave as is").tag(Double?.none)
            Text("1 second").tag(Double?.some(1))
            Text("2 seconds").tag(Double?.some(2))
            Text("4 seconds").tag(Double?.some(4))
        }
        HStack {
            Text("Bitrate (kbps)")
            Spacer()
            TextField("Leave as is", text: $bitrateText)
                .keyboardType(.numberPad)
                .multilineTextAlignment(.trailing)
                .frame(maxWidth: 140)
        }
        .onAppear { bitrateText = profile.bitrateKbps.map { "\($0)" } ?? "" }
        .onChange(of: bitrateText) { _, text in
            let digits = text.filter(\.isNumber)
            profile.bitrateKbps = digits.isEmpty ? nil : Int(digits)
        }
        .onChange(of: profile.bitrateKbps) { _, value in
            let text = value.map { "\($0)" } ?? ""
            if text != bitrateText.filter(\.isNumber) { bitrateText = text }
        }
    }
}

// MARK: - All cameras

struct MainStreamAllCamerasView: View {
    @EnvironmentObject private var session: SessionModel

    @State private var saved: MainStreamProfile?
    @State private var draft = MainStreamProfile.keep
    @State private var rows: [CameraMainStreamRow] = []
    @State private var cameras: [Camera] = []
    @State private var resetCameras = false
    @State private var busy = false
    @State private var results: [MainStreamResult]?
    @State private var loadError: String?
    @State private var actionError: String?

    var body: some View {
        Form {
            if let loadError {
                Section {
                    Text(loadError).font(.footnote).foregroundStyle(Theme.danger)
                }
                .listRowBackground(Theme.surface)
            }

            Section {
                MainStreamProfileFields(profile: $draft)
                Button("Use recommended: 1080p · H.264 · 1 s · 4096 kbps") {
                    draft = .recommended
                }
                .disabled(busy)
            } header: {
                Text("All cameras")
            } footer: {
                Text("The main stream is what is recorded, read for faces, and shown "
                     + "in fullscreen live view. \"Up to\" picks each camera's largest size at or "
                     + "under that height. Lowering the resolution does not lower the bitrate by "
                     + "itself — about 4096 kbps is plenty for 1080p H.264.")
                    .foregroundStyle(Theme.textSecondary)
            }
            .listRowBackground(Theme.surface)

            Section {
                let own = rows.filter { !$0.inherited }
                if !own.isEmpty {
                    Toggle(isOn: $resetCameras) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Also cameras with their own setting")
                                .foregroundStyle(Theme.textPrimary)
                            Text(own.map { friendly($0.camera) }.joined(separator: ", "))
                                .font(.caption2)
                                .foregroundStyle(Theme.textSecondary)
                        }
                    }
                    .tint(Theme.accent)
                    .disabled(busy)
                }
                Button {
                    Task { await run { try await $0.setMainStreamForAll(draft, resetCameras: resetCameras) } }
                } label: {
                    HStack {
                        Text(busy ? "Applying…" : "Apply to all cameras")
                        if busy { Spacer(); ProgressView() }
                    }
                }
                .disabled(busy || saved == nil || (draft == saved && !resetCameras))
                Button("Re-check cameras now") {
                    Task { await run { try await $0.reapplyMainStreams() } }
                }
                .disabled(busy)
            } footer: {
                Text("Changing resolution or codec restarts that camera's video for a few seconds; "
                     + "recording and live view reconnect on their own. Settings are re-applied "
                     + "when a camera reconnects and every 30 minutes.")
                    .foregroundStyle(Theme.textSecondary)
            }
            .listRowBackground(Theme.surface)

            Section("Cameras") {
                ForEach(rows, id: \.camera) { row in
                    let result = results?.first { $0.camera == row.camera } ?? row.last
                    VStack(alignment: .leading, spacing: 2) {
                        HStack {
                            Text(friendly(row.camera)).foregroundStyle(Theme.textPrimary)
                            Spacer()
                            Text(row.inherited ? "All cameras" : "Own setting")
                                .font(.caption)
                                .foregroundStyle(Theme.textSecondary)
                        }
                        if let result {
                            Text(result.summary)
                                .font(.caption2)
                                .foregroundStyle(result.ok || result.skipped == true
                                                 ? Theme.textSecondary : Theme.danger)
                        }
                    }
                }
            }
            .listRowBackground(Theme.surface)
        }
        .scrollContentBackground(.hidden)
        .background(Theme.bg)
        .navigationTitle("Video quality")
        .navigationBarTitleDisplayMode(.inline)
        .task { await load() }
        .refreshable { await load() }
        .alert(
            "Video quality",
            isPresented: Binding(get: { actionError != nil }, set: { if !$0 { actionError = nil } })
        ) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(actionError ?? "")
        }
    }

    private func friendly(_ name: String) -> String {
        cameras.first { $0.name == name }?.friendlyName ?? name
    }

    private func load() async {
        guard session.isAdmin, let api = session.api else {
            loadError = "Video quality is available to administrators only."
            return
        }
        do {
            async let o = api.mainStreamOverview()
            async let c = api.cameras()
            let overview = try await o
            cameras = (try? await c) ?? cameras
            saved = overview.profile
            if !busy { draft = overview.profile }
            rows = overview.cameras
            loadError = nil
        } catch {
            loadError = (error as? ApiError)?.message ?? error.localizedDescription
        }
    }

    private func run(_ call: (APIClient) async throws -> MainStreamResults) async {
        guard session.isAdmin, let api = session.api, !busy else { return }
        busy = true
        results = nil
        do {
            results = try await call(api).results
            resetCameras = false
        } catch {
            actionError = (error as? ApiError)?.message ?? error.localizedDescription
        }
        busy = false
        await load()
    }
}

// MARK: - One camera

struct CameraMainStreamView: View {
    let camera: Camera

    @EnvironmentObject private var session: SessionModel

    @State private var detail: CameraMainStreamDetail?
    @State private var followAll = true
    @State private var draft = MainStreamProfile.keep
    @State private var busy = false
    @State private var loading = true
    @State private var result: MainStreamResult?
    @State private var actionError: String?

    var body: some View {
        Form {
            Section {
                if loading {
                    HStack { ProgressView(); Text("Reading the camera…").foregroundStyle(Theme.textSecondary) }
                } else if let now = detail?.live?.current {
                    Text(now.summary)
                        .font(.footnote)
                        .foregroundStyle(Theme.textPrimary)
                    if let sub = detail?.live?.substream {
                        Text(sub.summary + (sub.isH264 ? "" :
                            " — live video cannot play until this is H.264. Vigilume sets it "
                            + "automatically when the camera connects (and every 30 minutes)."))
                            .font(.caption)
                            .foregroundStyle(sub.isH264 ? Theme.textSecondary : Theme.danger)
                    }
                } else {
                    Text("Could not read the camera" + (detail?.error.map { ": \($0)" } ?? "."))
                        .font(.footnote)
                        .foregroundStyle(Theme.danger)
                }
            } header: {
                Text("Now")
            }
            .listRowBackground(Theme.surface)

            Section {
                Picker("Setting", selection: $followAll) {
                    Text("Same as all cameras").tag(true)
                    Text("This camera's own").tag(false)
                }
                .pickerStyle(.segmented)
                .onChange(of: followAll) { _, follow in
                    if follow, let global = detail?.global { draft = global }
                }
                if followAll {
                    Text(detail?.global?.summary ?? "…")
                        .font(.footnote)
                        .foregroundStyle(Theme.textSecondary)
                } else {
                    MainStreamProfileFields(profile: $draft, sizes: detail?.live?.resolutions ?? [])
                }
            } footer: {
                Text("Give a camera that has to recognise faces from a distance a higher "
                     + "resolution than the rest — at 1080p a distant face has half the pixels "
                     + "it has at 4K.")
                    .foregroundStyle(Theme.textSecondary)
            }
            .listRowBackground(Theme.surface)

            Section {
                Button {
                    Task { await save() }
                } label: {
                    HStack {
                        Text(busy ? "Applying…" : "Apply to this camera")
                        if busy { Spacer(); ProgressView() }
                    }
                }
                .disabled(busy || loading)
                if let result {
                    Text(result.summary)
                        .font(.caption)
                        .foregroundStyle(result.ok || result.skipped == true
                                         ? Theme.textSecondary : Theme.danger)
                }
            } footer: {
                Text("Applied straight away. A resolution or codec change restarts the camera's "
                     + "video for a few seconds.")
                    .foregroundStyle(Theme.textSecondary)
            }
            .listRowBackground(Theme.surface)
        }
        .scrollContentBackground(.hidden)
        .background(Theme.bg)
        .navigationTitle("Video quality")
        .navigationBarTitleDisplayMode(.inline)
        .task { await load() }
        .refreshable { await load() }
        .alert(
            "Video quality",
            isPresented: Binding(get: { actionError != nil }, set: { if !$0 { actionError = nil } })
        ) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(actionError ?? "")
        }
    }

    private func load() async {
        guard session.isAdmin, let api = session.api else { loading = false; return }
        loading = true
        do {
            let d = try await api.cameraMainStream(camera.name)
            detail = d
            followAll = d.inherited
            draft = d.inherited ? (d.global ?? .keep) : (d.own ?? .keep)
        } catch {
            actionError = (error as? ApiError)?.message ?? error.localizedDescription
        }
        loading = false
    }

    private func save() async {
        guard session.isAdmin, let api = session.api, !busy else { return }
        busy = true
        result = nil
        do {
            let d = try await api.setCameraMainStream(camera.name, profile: followAll ? nil : draft)
            result = d.result
        } catch {
            actionError = (error as? ApiError)?.message ?? error.localizedDescription
        }
        busy = false
        await load()
    }
}
