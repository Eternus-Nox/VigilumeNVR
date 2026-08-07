import Foundation

// Camera device-control models mirroring docs/CONTRACTS.md
// (GET/PUT /api/cameras/{name}/settings, POST .../light|siren|reboot|probe).
// Reference client: frontend/src/lib/api.ts (DeviceSettings / ProbeResult).

// MARK: - Modes

/// IR illuminator mode (`ir_mode` in device settings).
enum IRMode: String, CaseIterable, Identifiable, Sendable {
    case auto
    case on
    case off

    var id: String { rawValue }

    var label: String {
        switch self {
        case .auto: return "Auto"
        case .on: return "On"
        case .off: return "Off"
        }
    }
}

/// White-light (spotlight) mode for POST /api/cameras/{name}/light.
/// Case order matches `IRMode` so both segmented pickers read Auto|On|Off.
enum SpotlightMode: String, CaseIterable, Identifiable, Sendable {
    case auto
    case on
    case off

    var id: String { rawValue }

    var label: String {
        switch self {
        case .auto: return "Auto"
        case .on: return "On"
        case .off: return "Off"
        }
    }
}

/// Night-vision mode (`night_vision_mode` in device settings) for cameras with
/// switchable night vision (caps.night_vision — the IP2M-1056E). `auto` lets
/// the camera decide, `color` forces full-color white-LED night vision, `bw`
/// forces classic infrared. Replaces the IR/day-night control on these cameras
/// (both drive the same Dahua day/night table).
enum NightVisionMode: String, CaseIterable, Identifiable, Sendable {
    case auto
    case color
    case bw

    var id: String { rawValue }

    var label: String {
        switch self {
        case .auto: return "Auto"
        case .color: return "Full-color"
        case .bw: return "IR"
        }
    }
}

// MARK: - Device settings (Amcrest state)

struct WhiteLightState: Codable, Sendable, Equatable {
    var mode: String        // "off" | "on" | "auto"
    var brightness: Int     // 0-100; meaningful when mode == "on"
}

/// Fields are present only when the capability exists on the device.
struct DeviceSettings: Codable, Sendable, Equatable {
    struct Volume: Codable, Sendable, Equatable {
        var mic: Int?
        var speaker: Int?
    }

    var irMode: String?
    /// "auto" | "color" | "bw"; present only on caps.night_vision cameras.
    var nightVisionMode: String?
    var whiteLight: WhiteLightState?
    var flip: Bool?
    var osdName: String?
    var motionDetect: Bool?
    var volume: Volume?
}

/// Sparse patch for PUT /api/cameras/{name}/settings — nil fields are
/// omitted from the JSON, so only the changed knob is applied. Mirrors the
/// backend `DeviceSettingsPatch` (routers/cameras.py): every field is additive
/// and capability-gated device-side.
struct DeviceSettingsPatch: Encodable, Sendable {
    struct WhiteLightPatch: Encodable, Sendable {
        var mode: String?
        var brightness: Int?
    }

    struct VolumePatch: Encodable, Sendable {
        var mic: Int?
        var speaker: Int?
    }

    var irMode: String?
    var nightVisionMode: String?
    var whiteLight: WhiteLightPatch?
    /// Flip the image 180° (VideoInOptions mirror/flip).
    var flip: Bool?
    /// On-screen display channel-title text.
    var osdName: String?
    /// On-camera (device-side) motion detection enable.
    var motionDetect: Bool?
    /// Mic / speaker volume (0–100). Backend applies speaker; mic is ignored on
    /// models without a verified CGI, so sending it is harmless.
    var volume: VolumePatch?
}

// MARK: - Probe

/// Result of POST /api/cameras/{name}/probe (getDeviceType + capability probe).
struct ProbeResult: Decodable, Sendable {
    let ok: Bool
    let model: String?
    let capabilities: CameraCapabilities
    /// "authentication failed" / "camera unreachable" on failure.
    let detail: String?
}
