import Foundation

// MARK: - ApiError

/// Mirrors frontend/src/lib/api.ts ApiError: an HTTP status plus a
/// human-readable message, with FastAPI `detail` flattening (string,
/// {message}, or validation-error arrays -> "field: msg; field: msg").
struct ApiError: Error, LocalizedError, Sendable {
    let status: Int          // 0 == network / transport error
    let message: String

    var errorDescription: String? { message }
    var isUnauthorized: Bool { status == 401 }

    static func network(_ underlying: Error? = nil) -> ApiError {
        ApiError(status: 0, message: "Network error — is the NVR reachable?")
    }

    /// Build from a non-2xx response body.
    static func from(status: Int, data: Data?) -> ApiError {
        var detail = "HTTP \(status)"
        if let data,
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            if let s = obj["detail"] as? String {
                detail = s
            } else if let s = obj["message"] as? String {
                detail = s
            } else if let arr = obj["detail"] as? [[String: Any]] {
                // FastAPI validation errors: [{loc:[...], msg, type}]
                let parts: [String] = arr.compactMap { entry in
                    var msg = (entry["msg"] as? String) ?? ""
                    if msg.hasPrefix("Value error, ") {
                        msg = String(msg.dropFirst("Value error, ".count))
                    }
                    let field = (entry["loc"] as? [Any])?.last.map { "\($0)" }
                    if let field, !msg.isEmpty { return "\(field): \(msg)" }
                    return msg.isEmpty ? nil : msg
                }
                if !parts.isEmpty { detail = parts.joined(separator: "; ") }
            }
        }
        return ApiError(status: status, message: detail)
    }
}

// MARK: - APIClient

/// Stateless async/await client for the Vigilume backend (docs/CONTRACTS.md).
/// One instance per (server, token) pair — SessionModel rebuilds it on login/
/// logout/server switch. All JSON is snake_case; decoding converts to
/// camelCase. Media URLs (AVPlayer/AsyncImage fetch these without our
/// headers) ALWAYS use the `?token=` form via `mediaURL(_:)`.
///
/// **Two base URLs — split by concern (per-server LAN routing).**
/// A saved server can carry an optional LAN address in addition to its primary
/// (usually HTTPS) URL. The backend + JWT are identical on both hosts — only the
/// host differs — so tokens work either way. We route accordingly:
///
///   - `apiBase` — the PRIMARY (usually HTTPS) URL. Used for ALL control / auth
///     / list traffic: login, `me`, cameras, events, groups, settings,
///     camera-control POSTs, APNs register/unregister, and BOTH
///     WebSockets (`/api/ws`, camera `/talk`). These always go over the primary
///     so nothing breaks when the LAN is absent/unreachable.
///   - `mediaBase` — the LAN URL when it's currently reachable, otherwise it's
///     just `apiBase`. Used ONLY for streaming / media URLs: go2rtc live
///     (WHEP + HLS main/sub), recording playlist + export, event clip/snapshot,
///     and camera snapshots. On LAN this gives a fast, direct, low-latency
///     path; off-LAN it collapses to the primary URL automatically.
///
/// The `mediaBase` choice is made by `SessionModel` from `LANReachability`; a
/// mid-session network change rebuilds the client with a new `mediaBase`, and
/// the live players re-resolve their URLs on the next attach.
struct APIClient: Sendable {
    /// Primary base for control/auth/list calls + WebSockets.
    let apiBase: URL
    /// Base for streaming/media URLs (LAN when reachable, else == `apiBase`).
    let mediaBase: URL
    let token: String?

    private let session: URLSession

    /// - Parameter mediaBase: pass the LAN URL to route media there; omit (nil)
    ///   to route media over `apiBase` too (the off-LAN / no-LAN default).
    init(apiBase: URL, mediaBase: URL? = nil, token: String?, session: URLSession = .shared) {
        self.apiBase = apiBase
        self.mediaBase = mediaBase ?? apiBase
        self.token = token
        self.session = session
    }

    private static let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.keyDecodingStrategy = .convertFromSnakeCase
        return d
    }()

    private static let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.keyEncodingStrategy = .convertToSnakeCase
        return e
    }()

    // MARK: Request plumbing

    private func makeRequest(
        _ method: String,
        _ path: String,
        query: [URLQueryItem] = [],
        body: Data? = nil,
        auth: Bool = true
    ) throws -> URLRequest {
        guard var components = URLComponents(
            url: apiBase.appendingPathComponent(path),   // control/auth: primary
            resolvingAgainstBaseURL: false
        ) else { throw ApiError(status: 0, message: "Invalid server URL") }
        if !query.isEmpty { components.queryItems = query }
        guard let url = components.url else {
            throw ApiError(status: 0, message: "Invalid request URL")
        }
        var request = URLRequest(url: url)
        request.httpMethod = method
        if auth, let token {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        if let body {
            request.httpBody = body
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }
        return request
    }

    @discardableResult
    private func send(_ request: URLRequest) async throws -> Data {
        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: request)
        } catch {
            throw ApiError.network(error)
        }
        guard let http = response as? HTTPURLResponse else {
            throw ApiError.network()
        }
        guard (200 ..< 300).contains(http.statusCode) else {
            throw ApiError.from(status: http.statusCode, data: data)
        }
        return data
    }

    private func get<T: Decodable>(_ path: String, query: [URLQueryItem] = []) async throws -> T {
        let data = try await send(try makeRequest("GET", path, query: query))
        return try Self.decoder.decode(T.self, from: data)
    }

    private func sendJSON<T: Decodable, B: Encodable>(
        _ method: String, _ path: String, body: B, auth: Bool = true
    ) async throws -> T {
        let request = try makeRequest(
            method, path, body: try Self.encoder.encode(body), auth: auth
        )
        let data = try await send(request)
        return try Self.decoder.decode(T.self, from: data)
    }

    // MARK: Media URL helper

    /// Append `?token=` to an API path and resolve it against `mediaBase` (the
    /// LAN host when reachable, else the primary URL). Use for anything fetched
    /// by AVPlayer / the OS image loader (playlists, segments, snapshots,
    /// clips) — mirrors api.ts `mediaUrl`.
    func mediaURL(_ path: String, extraQuery: [URLQueryItem] = []) -> URL {
        var components = URLComponents(
            url: mediaBase.appendingPathComponent(path),   // media: LAN-or-primary
            resolvingAgainstBaseURL: false
        )!
        var items = extraQuery
        if let token {
            items.append(URLQueryItem(name: "token", value: token))
        }
        if !items.isEmpty { components.queryItems = items }
        return components.url!
    }

    // MARK: Auth

    /// POST /api/auth/login — no Bearer needed.
    func login(username: String, password: String) async throws -> LoginResponse {
        struct Body: Encodable { let username: String; let password: String }
        return try await sendJSON(
            "POST", "api/auth/login",
            body: Body(username: username, password: password), auth: false
        )
    }

    /// GET /api/auth/me — validates a stored session at launch.
    func me() async throws -> MeResponse {
        try await get("api/auth/me")
    }

    // MARK: Cameras

    func cameras() async throws -> [Camera] {
        try await get("api/cameras")
    }

    /// PUT /api/cameras/{name} changing ONLY the server-detection mode. The
    /// update contract (CameraUpdate) requires the identity fields
    /// (name/friendly_name/model/ip) on every call, so we echo them back
    /// verbatim and send no credentials — a blank username/password tells the
    /// backend to keep the stored ones. Every other editable field is omitted,
    /// so it keeps its stored value; only `detect_mode` changes. Returns the
    /// updated camera (with the effective mode + live ai_active). Admin-only.
    func setCameraDetectMode(_ camera: Camera, mode: DetectMode) async throws -> Camera {
        struct Body: Encodable {
            let name: String
            let friendlyName: String
            let model: String
            let ip: String
            let detectMode: String
        }
        return try await sendJSON(
            "PUT", "api/cameras/\(camera.name)",
            body: Body(
                name: camera.name,
                friendlyName: camera.friendlyName,
                model: camera.model,
                ip: camera.ip,
                detectMode: mode.rawValue
            )
        )
    }

    /// PUT /api/cameras/{name} — full per-camera config edit (CameraUpdate).
    /// Identity fields (name/friendly_name/model/ip) are required on every call;
    /// a blank username/password keeps the stored credentials. Every OTHER field
    /// is optional and OMITTED-when-nil (synthesized `encodeIfPresent`), which
    /// the backend reads as "keep the stored value" — so a caller sends only the
    /// concerns it's editing. Returns the updated camera. Admin-only.
    @discardableResult
    func updateCamera(_ payload: CameraUpdatePayload) async throws -> Camera {
        try await sendJSON("PUT", "api/cameras/\(payload.name)", body: payload)
    }

    /// POST /api/cameras — adopt a new camera (CameraInput). The backend
    /// validates the name slug, model and IP, probes the device's capabilities,
    /// regenerates the go2rtc config and reloads the detection/recording
    /// engines, then returns the created camera (201). A duplicate name is a
    /// 409 with a readable detail. Admin-only.
    @discardableResult
    func addCamera(_ payload: CameraCreatePayload) async throws -> Camera {
        try await sendJSON("POST", "api/cameras", body: payload)
    }

    /// DELETE /api/cameras/{name} — forget a camera (204). Its recordings and
    /// events stay on disk until retention cleanup; only the camera config goes.
    /// The streams reload, so live view drops briefly. Admin-only.
    func deleteCamera(name: String) async throws {
        try await send(try makeRequest("DELETE", "api/cameras/\(escapePath(name))"))
    }

    /// PUT /api/cameras/order {names} — set the dashboard camera order. The list
    /// may be full or partial: names not listed keep their relative order after
    /// the listed ones, and unknown names are ignored. Purely cosmetic (touches
    /// only the position column). Returns the reordered camera list. Admin-only.
    @discardableResult
    func setCameraOrder(_ names: [String]) async throws -> [Camera] {
        struct Body: Encodable { let names: [String] }
        return try await sendJSON("PUT", "api/cameras/order", body: Body(names: names))
    }

    /// Percent-encode a path segment (camera names are strict lowercase slugs by
    /// contract, so this is normally a no-op — it just keeps a hand-typed or
    /// legacy name from breaking the URL).
    private func escapePath(_ value: String) -> String {
        value.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? value
    }

    /// GET /api/detection/labels — the active detector's selectable label
    /// vocabulary, for the per-camera object picker. Admin-only; may 404 on an
    /// older backend (callers fall back to a bundled list).
    func detectionLabels() async throws -> LabelsResponse {
        try await get("api/detection/labels")
    }

    /// Live snapshot poll (Bearer-free media URL for AsyncImage).
    func cameraSnapshotURL(_ name: String) -> URL {
        mediaURL("api/cameras/\(name)/snapshot.jpg")
    }

    // MARK: Live streams (go2rtc HLS/fMP4 — unauthenticated proxy, §2 of ios-design.md)

    /// Full-res live stream with AAC audio (camera detail).
    func liveStreamURL(camera: String) -> URL {
        go2rtcStreamURL(src: camera)
    }

    /// Low-res video-only substream (muted grid tiles).
    func liveSubStreamURL(camera: String) -> URL {
        go2rtcStreamURL(src: "\(camera)_sub")
    }

    private func go2rtcStreamURL(src: String) -> URL {
        var components = URLComponents(
            url: mediaBase.appendingPathComponent("go2rtc/api/stream.m3u8"),  // media
            resolvingAgainstBaseURL: false
        )!
        // NO token param. It was added for an nginx auth_request gate on the
        // go2rtc handshakes; that gate has been reverted, so the token now buys
        // nothing here — and go2rtc parses this query string to choose the
        // output media, where an unrecognised key is a needless risk.
        components.queryItems = [
            URLQueryItem(name: "src", value: src),
            URLQueryItem(name: "mp4", value: nil),   // fMP4 flavor: H.264+HEVC+AAC
        ]
        return components.url!
    }

    // MARK: Live streams (go2rtc WebRTC/WHEP — sub-second live, §2 of ios-design.md)
    //
    // WHEP endpoint verified against the pinned go2rtc v1.9.14 source
    // (internal/webrtc/{webrtc,server}.go): the ONLY WebRTC HTTP route is
    // `api/webrtc` (there is NO `api/whep` route in this version). With
    // `Content-Type: application/sdp` and method POST, `outputWebRTC` treats the
    // raw body as the SDP offer, runs the WHEP exchange, and returns `201
    // Created` with `Content-Type: application/sdp` and the answer as the body.
    // Same unauthenticated `/go2rtc/` proxy as the HLS paths.

    /// WHEP endpoint for the full-res main stream (camera detail / full-screen).
    func liveStreamWHEPURL(camera: String) -> URL {
        go2rtcWHEPURL(src: camera)
    }

    /// WHEP endpoint for the low-res video substream (grid tiles).
    func liveSubStreamWHEPURL(camera: String) -> URL {
        go2rtcWHEPURL(src: "\(camera)_sub")
    }

    private func go2rtcWHEPURL(src: String) -> URL {
        var components = URLComponents(
            url: mediaBase.appendingPathComponent("go2rtc/api/webrtc"),  // media
            resolvingAgainstBaseURL: false
        )!
        // No token param — see go2rtcStreamURL: the gate it served is reverted.
        components.queryItems = [URLQueryItem(name: "src", value: src)]
        return components.url!
    }


    // MARK: Events

    func events(
        camera: String? = nil,
        label: String? = nil,
        after: Double? = nil,
        before: Double? = nil,
        limit: Int = 50,
        offset: Int = 0
    ) async throws -> EventsPage {
        var query: [URLQueryItem] = [
            URLQueryItem(name: "limit", value: String(limit)),
            URLQueryItem(name: "offset", value: String(offset)),
        ]
        if let camera { query.append(URLQueryItem(name: "camera", value: camera)) }
        if let label { query.append(URLQueryItem(name: "label", value: label)) }
        if let after { query.append(URLQueryItem(name: "after", value: String(after))) }
        if let before { query.append(URLQueryItem(name: "before", value: String(before))) }
        return try await get("api/events", query: query)
    }

    func event(id: Int) async throws -> EventDetail {
        try await get("api/events/\(id)")
    }

    func deleteEvent(id: Int) async throws {
        try await send(try makeRequest("DELETE", "api/events/\(id)"))
    }

    /// ADMIN: permanently delete ALL events plus their snapshots and clips.
    /// Continuous recordings are kept. Irreversible.
    func deleteAllEvents() async throws {
        try await send(try makeRequest("DELETE", "api/events"))
    }

    /// ADMIN: permanently delete ALL continuous recorded footage for every
    /// camera. Events and their clips are kept. Irreversible.
    func deleteAllRecordings() async throws {
        try await send(try makeRequest("DELETE", "api/recordings"))
    }

    func eventSnapshotURL(id: Int) -> URL {
        mediaURL("api/events/\(id)/snapshot.jpg")
    }

    func eventClipURL(id: Int) -> URL {
        mediaURL("api/events/\(id)/clip.mp4")
    }

    // MARK: Suppressions (reject-to-suppress; admin-only)

    /// ADMIN: POST /api/events/{id}/reject — mark the event a false detection,
    /// learn a suppression, and delete the event. The created suppression is
    /// returned in the body; we ignore it here (the Excluded-objects list
    /// refetches). The deleted event's row is dropped via .vigilumeEventDeleted.
    func rejectEvent(id: Int) async throws {
        try await send(try makeRequest("POST", "api/events/\(id)/reject"))
    }

    /// ADMIN: GET /api/detection/suppressions — learned suppressions, newest
    /// first (the Settings › Excluded objects list).
    func suppressions() async throws -> [Suppression] {
        try await get("api/detection/suppressions")
    }

    /// ADMIN: DELETE /api/detection/suppressions/{id} — forget a suppression
    /// (idempotent 204).
    func deleteSuppression(id: Int) async throws {
        try await send(try makeRequest("DELETE", "api/detection/suppressions/\(id)"))
    }

    /// Suppression thumbnail (Bearer-free media URL for AsyncImage). 404 when the
    /// suppression has no thumb — AsyncImage falls back to its placeholder.
    func suppressionThumbURL(id: Int) -> URL {
        mediaURL("api/detection/suppressions/\(id)/thumb.jpg")
    }

    // MARK: Recordings

    func recordingCameras() async throws -> [RecordingCamera] {
        try await get("api/recordings/cameras")
    }

    /// date is a LOCAL day "YYYY-MM-DD" (server tz).
    func recordingIndex(camera: String, date: String) async throws -> RecordingIndex {
        try await get(
            "api/recordings/\(camera)/index",
            query: [URLQueryItem(name: "date", value: date)]
        )
    }

    /// HLS VOD playlist for AVPlayer (window capped server-side at 6 h).
    func recordingPlaylistURL(camera: String, start: Double, end: Double) -> URL {
        mediaURL(
            "api/recordings/\(camera)/playlist.m3u8",
            extraQuery: [
                URLQueryItem(name: "start", value: String(Int(start))),
                URLQueryItem(name: "end", value: String(Int(end))),
            ]
        )
    }

    /// One downloadable MP4 (window capped server-side at 30 min).
    func recordingExportURL(camera: String, start: Double, end: Double) -> URL {
        mediaURL(
            "api/recordings/\(camera)/export.mp4",
            extraQuery: [
                URLQueryItem(name: "start", value: String(Int(start))),
                URLQueryItem(name: "end", value: String(Int(end))),
            ]
        )
    }

    // MARK: Software Privacy Mode (ADMIN-ONLY, both verbs)

    /// GET /api/privacy — the per-camera / per-group capture kill switch.
    ///
    /// **Admin-only; a viewer gets 403.** Call this ONLY from an admin-gated
    /// screen. The dashboard must NOT use it to decide whether to show the
    /// Privacy Mode overlay — that comes from `Camera.isPrivate`, which every
    /// authenticated user receives on GET /api/cameras.
    func privacyMode() async throws -> PrivacyModeState {
        try await get("api/privacy")
    }

    /// POST /api/privacy — set which cameras/groups are private (admin-only).
    ///
    /// PARTIAL by contract: a nil field is left unchanged server-side, so pass
    /// only what you are changing. The response is the authoritative resolved
    /// state — adopt it rather than optimistically toggling locally, or a
    /// camera that went private via a GROUP will render wrong.
    @discardableResult
    func setPrivacyMode(cameras: [String]? = nil, groups: [Int]? = nil) async throws
        -> PrivacyModeState
    {
        struct Body: Encodable { let cameras: [String]?; let groups: [Int]? }
        return try await sendJSON(
            "POST", "api/privacy", body: Body(cameras: cameras, groups: groups)
        )
    }

    // MARK: Groups (full CRUD for both roles)

    func groups() async throws -> [CameraGroup] {
        try await get("api/groups")
    }

    func createGroup(name: String, cameras: [String]) async throws -> CameraGroup {
        struct Body: Encodable { let name: String; let cameras: [String] }
        return try await sendJSON("POST", "api/groups", body: Body(name: name, cameras: cameras))
    }

    func updateGroup(
        id: Int, name: String? = nil, cameras: [String]? = nil, position: Int? = nil
    ) async throws -> CameraGroup {
        struct Body: Encodable { let name: String?; let cameras: [String]?; let position: Int? }
        return try await sendJSON(
            "PUT", "api/groups/\(id)",
            body: Body(name: name, cameras: cameras, position: position)
        )
    }

    func deleteGroup(id: Int) async throws {
        try await send(try makeRequest("DELETE", "api/groups/\(id)"))
    }

    // MARK: Users (admin-only CRUD; viewers get 403)
    //
    // The built-in admin is env-controlled and has no DB row: it never appears
    // in `users()` and cannot be created, updated or deleted here. The backend
    // owns every rule (reserved "admin" name, the username regex, the 8-char
    // password floor, "cannot demote the last admin") and returns each as a
    // readable `detail` — callers surface `ApiError.message` rather than
    // re-implementing the policy.

    /// GET /api/users — the additional accounts, `{id, username, role, created_at}`.
    func users() async throws -> [ManagedUser] {
        try await get("api/users")
    }

    /// POST /api/users — create an account. 400 on a reserved/invalid username
    /// or a short password, 409 if the username is taken.
    func createUser(username: String, password: String, role: Role) async throws -> ManagedUser {
        struct Body: Encodable { let username: String; let password: String; let role: String }
        return try await sendJSON(
            "POST", "api/users",
            body: Body(username: username, password: password, role: role.rawValue)
        )
    }

    /// PUT /api/users/{id} — reset the password and/or change the role. Both are
    /// optional and omitted-when-nil (synthesized `encodeIfPresent`), which the
    /// backend reads as "leave it alone", so a caller sends only what it edited.
    /// Demoting the last remaining DB admin is a 400.
    @discardableResult
    func updateUser(id: Int, password: String? = nil, role: Role? = nil) async throws -> ManagedUser {
        struct Body: Encodable { let password: String?; let role: String? }
        return try await sendJSON(
            "PUT", "api/users/\(id)", body: Body(password: password, role: role?.rawValue)
        )
    }

    /// DELETE /api/users/{id} — remove an account (204).
    func deleteUser(id: Int) async throws {
        try await send(try makeRequest("DELETE", "api/users/\(id)"))
    }

    // MARK: Settings (admin-only routes; viewers get 403)

    /// GET /api/settings — the full settings document, of which we decode only
    /// the detection + system blocks (see `SettingsDocument`). The response also
    /// carries a computed read-only `webrtc` block; we neither read nor echo it.
    func settingsDocument() async throws -> SettingsDocument {
        try await get("api/settings")
    }

    /// PATCH /api/settings — deep-merge a PARTIAL settings document. Returns the
    /// updated document.
    ///
    /// **This is the ONLY way the app writes settings — never PUT.** PUT is a
    /// full-document replace and every field has a backend default, so a key the
    /// body omits is RESET rather than preserved: a PUT without
    /// `notifications.apns.direct.p8` destroys the user's APNs signing key and
    /// breaks push (verified empirically). PATCH touches only the keys sent, so
    /// a caller passes just the subtree it edited, e.g.
    /// `SettingsPatch(system: .init(publicUrl: …, webrtcCandidates: […]))`.
    /// Validation and every side-effect (detector reconfigure/model activation,
    /// go2rtc regen, MQTT restart) are identical to PUT; invalid values are a
    /// readable 422. Admin-only.
    @discardableResult
    func patchSettings(_ patch: SettingsPatch) async throws -> SettingsDocument {
        try await sendJSON("PATCH", "api/settings", body: patch)
    }

    // MARK: Detection models (read; admin-only)

    func detectionModels() async throws -> DetectionModelsResponse {
        try await get("api/detection/models")
    }

    /// ADMIN: POST /api/detection/models/{key}/download (202) — fetch a model's
    /// weights. Progress is observed by re-polling detectionModels() (state).
    func downloadModel(key: String) async throws {
        try await send(try makeRequest("POST", "api/detection/models/\(key)/download"))
    }

    /// ADMIN: DELETE /api/detection/models/{key} — remove a downloaded model's
    /// file. 409 if it is the active model ("switch tier first").
    func deleteModel(key: String) async throws {
        try await send(try makeRequest("DELETE", "api/detection/models/\(key)"))
    }

    // MARK: System

    func systemHealth() async throws -> SystemHealth {
        try await get("api/system/health")
    }

    /// ADMIN: GET /api/system/detector — detector self-test + per-camera ingest.
    func detector() async throws -> DetectorStatus {
        try await get("api/system/detector")
    }

    /// GET /api/system/camera-health?hours= — RTSP-port reachability over a
    /// window (require_auth, so viewer-visible). hours = 24 / 168 / 720.
    func cameraHealth(hours: Int) async throws -> CameraHealthReport {
        try await get("api/system/camera-health",
                      query: [URLQueryItem(name: "hours", value: String(hours))])
    }

    /// ADMIN: GET /api/integrations/rclone/providers — the storage backends the
    /// server can configure, with the fields each needs.
    func rcloneProviders() async throws -> [RcloneProvider] {
        struct Wrapper: Decodable { let providers: [RcloneProvider] }
        let w: Wrapper = try await get("api/integrations/rclone/providers")
        return w.providers
    }

    /// ADMIN: GET /api/integrations/rclone/remotes — configured destinations.
    /// Secrets are redacted server-side and never reach this device.
    func rcloneRemotes() async throws -> RcloneRemotesResponse {
        try await get("api/integrations/rclone/remotes")
    }

    /// ADMIN: POST /api/integrations/rclone/remotes — create (or replace) one,
    /// then probe it.
    func createRcloneRemote(
        name: String, type: String, values: [String: String]
    ) async throws -> RcloneCreateResult {
        struct Body: Encodable {
            let name: String
            let type: String
            let values: [String: String]
        }
        // NOTE the explicit CodingKeys-free shape: `values` is a free-form map
        // whose keys are rclone's own (access_key_id, secret_access_key…). The
        // encoder's .convertToSnakeCase applies to PROPERTY names, not to
        // dictionary keys, so these reach the server exactly as typed.
        return try await sendJSON(
            "POST", "api/integrations/rclone/remotes",
            body: Body(name: name, type: type, values: values)
        )
    }

    /// ADMIN: DELETE /api/integrations/rclone/remotes/{name} — forget the
    /// credentials. Deletes NOTHING in the cloud.
    func deleteRcloneRemote(name: String) async throws {
        let escaped = name.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? name
        try await send(try makeRequest("DELETE", "api/integrations/rclone/remotes/\(escaped)"))
    }

    /// ADMIN: POST /api/integrations/rclone/remotes/{name}/test.
    func testRcloneRemote(name: String) async throws -> RcloneTestResult {
        let escaped = name.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? name
        let data = try await send(
            try makeRequest("POST", "api/integrations/rclone/remotes/\(escaped)/test")
        )
        return try Self.decoder.decode(RcloneTestResult.self, from: data)
    }

    /// ADMIN: GET /api/integrations/archive/status — what the nightly cloud
    /// archive has actually done.
    func archiveStatus() async throws -> ArchiveStatus {
        try await get("api/integrations/archive/status")
    }

    /// ADMIN: POST /api/integrations/archive/run — run a pass NOW.
    ///
    /// Uses the SAVED settings, not a draft, and runs the real nightly pass
    /// rather than a separate probe — so a clean result is evidence the remote
    /// works. Deliberately slow: a day of clips on a thin uplink takes a while,
    /// so callers must show progress rather than assume it hung.
    func runArchive() async throws -> ArchiveRunResult {
        // makeRequest + decode rather than sendJSON: the route takes no body,
        // and sendJSON requires one to encode.
        let data = try await send(try makeRequest("POST", "api/integrations/archive/run"))
        return try Self.decoder.decode(ArchiveRunResult.self, from: data)
    }

    /// ADMIN: POST /api/integrations/mqtt/test — probe the broker with a DRAFT
    /// config (no save). Returns {ok, detail}.
    func testMqtt(enabled: Bool, host: String, port: Int, username: String,
                  password: String, discoveryPrefix: String,
                  baseTopic: String) async throws -> MqttTestResult {
        struct Cfg: Encodable {
            let enabled: Bool
            let host: String
            let port: Int
            let username: String
            let password: String
            let discoveryPrefix: String
            let baseTopic: String
        }
        struct Body: Encodable { let mqtt: Cfg }
        let body = Body(mqtt: Cfg(
            enabled: enabled, host: host, port: port, username: username,
            password: password, discoveryPrefix: discoveryPrefix, baseTopic: baseTopic))
        return try await sendJSON("POST", "api/integrations/mqtt/test", body: body)
    }

    /// POST /api/system/restart — ADMIN. Schedules a SIGTERM self-restart (202);
    /// the API is briefly unavailable while it comes back. Body is ignored.
    func restartServer() async throws {
        try await send(try makeRequest("POST", "api/system/restart"))
    }

    /// GET /api/notifications/apns/devices — registered APNs devices (8-char
    /// token prefixes only; require_auth). Lets a phone confirm it registered.
    func apnsDevices() async throws -> [ApnsDevice] {
        try await get("api/notifications/apns/devices")
    }

    /// POST /api/users/me/password {current_password, new_password} -> 204.
    /// Any authenticated DB user (the built-in env admin gets 400). new 8…256.
    func changeOwnPassword(current: String, new: String) async throws {
        struct Body: Encodable {
            let currentPassword: String
            let newPassword: String
            enum CodingKeys: String, CodingKey {
                case currentPassword = "current_password"
                case newPassword = "new_password"
            }
        }
        let body = try Self.encoder.encode(Body(currentPassword: current, newPassword: new))
        try await send(try makeRequest("POST", "api/users/me/password", body: body))
    }

    // MARK: Notifications (APNs registration — docs/push-architecture.md §2;
    // a 404 here means "server does not support native push yet": gate the UI.)

    /// POST /api/notifications/apns/register
    /// {device_token, device_name, key_b64, environment}. `keyB64` is the
    /// per-server 32-byte E2E key (PushCrypto); the server encrypts every
    /// push with it, the relay only ever sees ciphertext.
    func registerAPNs(
        deviceToken: String, deviceName: String, keyB64: String, environment: String
    ) async throws {
        struct Body: Encodable {
            let deviceToken: String
            let deviceName: String
            let keyB64: String
            let environment: String
            // Explicit keys: don't rely on snake-case digit-splitting for key_b64.
            enum CodingKeys: String, CodingKey {
                case deviceToken = "device_token"
                case deviceName = "device_name"
                case keyB64 = "key_b64"
                case environment
            }
        }
        let body = try Self.encoder.encode(Body(
            deviceToken: deviceToken, deviceName: deviceName,
            keyB64: keyB64, environment: environment
        ))
        try await send(try makeRequest("POST", "api/notifications/apns/register", body: body))
    }

    /// POST /api/push/voip {token, device_name, environment} — register this
    /// device's PushKit VoIP token so the backend can ring the phone (CallKit)
    /// on an AD410 doorbell press. Separate from the alert-push registration
    /// above: VoIP rides the same token-auth .p8 but a different APNs topic
    /// (<bundle>.voip) and push-type. The payload is minimal + NOT E2E-encrypted
    /// (the app must read it immediately to report the call), so no key is sent.
    /// May 404 on a backend that predates the feature — callers treat that as
    /// "server doesn't support VoIP calls yet" and simply skip.
    func registerVoIP(token: String, deviceName: String, environment: String) async throws {
        struct Body: Encodable {
            let token: String
            let deviceName: String
            let environment: String
            enum CodingKeys: String, CodingKey {
                case token
                case deviceName = "device_name"
                case environment
            }
        }
        let body = try Self.encoder.encode(Body(
            token: token, deviceName: deviceName, environment: environment
        ))
        try await send(try makeRequest("POST", "api/push/voip", body: body))
    }

    /// DELETE /api/notifications/apns/register {device_token} — the pinned
    /// contract's unregister (idempotent 204). NOT POST /apns/unregister:
    /// the backend only routes the DELETE.
    func unregisterAPNs(deviceToken: String) async throws {
        struct Body: Encodable { let deviceToken: String }
        let body = try Self.encoder.encode(Body(deviceToken: deviceToken))
        try await send(try makeRequest("DELETE", "api/notifications/apns/register", body: body))
    }

    // MARK: WebSocket URLs

    /// ws(s)://{base}/api/ws?token= — live event/camera/model updates.
    func websocketURL() -> URL? {
        // No ?token= — the JWT rides Sec-WebSocket-Protocol (wsSubprotocols).
        wsURL(path: "api/ws", query: [])
    }

    /// ws(s)://{base}/api/cameras/{name}/talk?token= — PTT uplink
    /// (binary Int16LE mono 8 kHz PCM frames; session JWT only).
    func talkWebSocketURL(camera: String) -> URL? {
        wsURL(path: "api/cameras/\(camera)/talk", query: [])
    }

    /// Subprotocols carrying the session JWT for a WebSocket handshake:
    /// `URLSession.shared.webSocketTask(with: url, protocols: wsSubprotocols())`.
    ///
    /// NOT a `?token=` query param. nginx writes the full request line — query
    /// string included — into its ERROR log, where `log_format` does not apply,
    /// so a tokened WS URL prints a live 30-day admin credential in cleartext
    /// on any warning. That happened once and cost a secret rotation. The
    /// server echoes "bearer" back on accept.
    func wsSubprotocols() -> [String] {
        guard let token else { return [] }
        return ["bearer", token]
    }

    private func tokenQuery() -> [URLQueryItem] {
        guard let token else { return [] }
        return [URLQueryItem(name: "token", value: token)]
    }

    private func wsURL(path: String, query: [URLQueryItem]) -> URL? {
        // WebSockets are control/live-event + PTT uplink → primary (apiBase).
        guard var components = URLComponents(
            url: apiBase.appendingPathComponent(path),
            resolvingAgainstBaseURL: false
        ) else { return nil }
        components.scheme = (components.scheme == "https") ? "wss" : "ws"
        if !query.isEmpty { components.queryItems = query }
        return components.url
    }
}
