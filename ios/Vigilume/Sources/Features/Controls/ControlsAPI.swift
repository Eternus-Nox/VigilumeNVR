import Foundation

// Camera-control endpoints (admin-only routes, docs/CONTRACTS.md):
//   GET/PUT /api/cameras/{name}/settings
//   POST    /api/cameras/{name}/light      {mode, brightness?}   (501 if firmware rejects)
//   POST    /api/cameras/{name}/siren      {duration_s?}         (501 if firmware rejects)
//   POST    /api/cameras/{name}/reboot
//   POST    /api/cameras/{name}/probe
//   POST    /api/cameras/{name}/ptz        {action, direction?, speed?, index?}
//                                          (501 not supported / 502 camera busy)
//
// APIClient's core plumbing is private to its file, so this extension carries
// its own minimal request helper built on the same public pieces (apiBase,
// token, ApiError) with identical semantics. These are camera-control commands
// (control traffic) → they ride `apiBase` (the primary URL), never the LAN.

extension APIClient {
    // MARK: Device settings

    func cameraDeviceSettings(_ name: String) async throws -> DeviceSettings {
        let data = try await controlsSend("GET", "api/cameras/\(name)/settings")
        return try ControlsJSON.decoder.decode(DeviceSettings.self, from: data)
    }

    @discardableResult
    func updateCameraDeviceSettings(
        _ name: String, patch: DeviceSettingsPatch
    ) async throws -> DeviceSettings {
        let body = try ControlsJSON.encoder.encode(patch)
        let data = try await controlsSend("PUT", "api/cameras/\(name)/settings", body: body)
        return try ControlsJSON.decoder.decode(DeviceSettings.self, from: data)
    }

    // MARK: One-shot controls

    func setLight(camera: String, mode: SpotlightMode, brightness: Int? = nil) async throws {
        struct Body: Encodable {
            let mode: String
            let brightness: Int?
        }
        let body = try ControlsJSON.encoder.encode(
            Body(mode: mode.rawValue, brightness: brightness)
        )
        _ = try await controlsSend("POST", "api/cameras/\(camera)/light", body: body)
    }

    func soundSiren(camera: String, durationS: Int = 10) async throws {
        struct Body: Encodable {
            let durationS: Int
        }
        let body = try ControlsJSON.encoder.encode(Body(durationS: durationS))
        _ = try await controlsSend("POST", "api/cameras/\(camera)/siren", body: body)
    }

    func rebootCamera(_ name: String) async throws {
        _ = try await controlsSend("POST", "api/cameras/\(name)/reboot")
    }

    func probeCamera(_ name: String) async throws -> ProbeResult {
        let data = try await controlsSend("POST", "api/cameras/\(name)/probe")
        return try ControlsJSON.decoder.decode(ProbeResult.self, from: data)
    }

    // MARK: PTZ

    /// POST /api/cameras/{name}/ptz — pan/tilt/zoom + presets on caps.ptz
    /// cameras (the IP3M-941B dome). `direction` is required for move/stop,
    /// `index` for the preset_* actions; the backend applies speed 4 by
    /// default. Surfaces 501 (not supported) / 502 (camera busy) as ApiError.
    func ptz(
        camera: String,
        action: PTZAction,
        direction: PTZDirection? = nil,
        speed: Int? = nil,
        index: Int? = nil
    ) async throws {
        struct Body: Encodable {
            let action: String
            let direction: String?
            let speed: Int?
            let index: Int?
        }
        let body = try ControlsJSON.encoder.encode(
            Body(
                action: action.rawValue,
                direction: direction?.rawValue,
                speed: speed,
                index: index
            )
        )
        _ = try await controlsSend("POST", "api/cameras/\(camera)/ptz", body: body)
    }
}

// MARK: - PTZ command vocabulary

/// PTZ verb for POST /api/cameras/{name}/ptz. Raw values are the on-the-wire
/// `action` strings. `step` nudges one small increment in `direction` (a single
/// self-contained move — no separate stop); the preset_* forms need an `index`.
enum PTZAction: String, Sendable {
    case step
    case move
    case stop
    case presetSet = "preset_set"
    case presetGoto = "preset_goto"
    case presetClear = "preset_clear"
}

/// The eight PTZ movement directions (four cardinal + four diagonal). Raw
/// values are the exact backend `direction` strings.
enum PTZDirection: String, Sendable, CaseIterable {
    case up, down, left, right
    case upleft, upright, downleft, downright
}

// MARK: - Private plumbing

private extension APIClient {
    func controlsSend(_ method: String, _ path: String, body: Data? = nil) async throws -> Data {
        var request = URLRequest(url: apiBase.appendingPathComponent(path))
        request.httpMethod = method
        if let token {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        if let body {
            request.httpBody = body
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }

        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await URLSession.shared.data(for: request)
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
}

enum ControlsJSON {
    static let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.keyDecodingStrategy = .convertFromSnakeCase
        return d
    }()

    static let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.keyEncodingStrategy = .convertToSnakeCase
        return e
    }()
}
