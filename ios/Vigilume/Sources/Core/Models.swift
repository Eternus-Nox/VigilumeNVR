import Foundation

// Typed models mirroring docs/CONTRACTS.md (reference client:
// frontend/src/lib/api.ts). JSON is snake_case; APIClient decodes with
// .convertFromSnakeCase, so property names below are the camelCase twins.

// MARK: - Auth

enum Role: String, Codable, Sendable {
    case admin
    case viewer

    /// Unknown role strings degrade to viewer (least privilege).
    init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode(String.self)
        self = Role(rawValue: raw) ?? .viewer
    }
}

struct LoginResponse: Decodable, Sendable {
    let token: String
    let role: Role
    let username: String
}

struct MeResponse: Decodable, Sendable {
    let username: String
    let role: Role
}

// MARK: - Cameras

/// How server-side detection is gated for a camera (mirrors
/// config.VALID_DETECT_MODES). `always` = continuous server inference;
/// `cameraAi` = server inference only while the camera's own AI fires;
/// `cameraAiOnly` = no server inference, events come straight from the camera
/// AI. A missing or unknown wire value degrades to `.always` — the backend
/// never silently disables detection, and neither do we.
enum DetectMode: String, Codable, Sendable, Hashable, CaseIterable {
    case always
    case cameraAi = "camera_ai"
    case cameraAiOnly = "camera_ai_only"

    init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode(String.self)
        self = DetectMode(rawValue: raw) ?? .always
    }

    /// Short label for the "where detection runs" status indicator.
    var runsLabel: String {
        switch self {
        case .always: return "Server"
        case .cameraAi: return "Camera-triggered"
        case .cameraAiOnly: return "On-camera"
        }
    }
}

struct CameraCapabilities: Decodable, Sendable, Hashable {
    let ir: Bool
    let whiteLight: Bool
    let siren: Bool
    let mic: Bool
    let speaker: Bool
    let doorbell: Bool
    let aiOnCamera: Bool
    /// True when the camera negotiates an RTSP/ONVIF audio *backchannel* (the
    /// AD410 doorbell): two-way talk must ride the go2rtc WebRTC mic uplink, not
    /// the HTTP-CGI `postAudio` path, which that firmware rejects. Absent on a
    /// backend that predates the feature — decode a missing value as `false` so
    /// those cameras keep the CGI talk fallback.
    let backchannel: Bool
    /// True when the camera exposes PTZ (pan/tilt/zoom) + presets (the
    /// IP3M-941B dome). Gates the directional pad + preset UI. Absent on a
    /// backend that predates the feature — decode a missing value as `false`.
    let ptz: Bool
    /// True when the camera has switchable night vision (auto / full-color
    /// white-LED / IR) — the IP2M-1056E. Gates the night-vision picker AND
    /// suppresses the old IR/day-night control (they drive the same Dahua
    /// day/night table and would clobber each other). Missing → `false`.
    let nightVision: Bool

    private enum CodingKeys: String, CodingKey {
        case ir, whiteLight, siren, mic, speaker, doorbell, aiOnCamera, backchannel
        case ptz, nightVision
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        ir = try c.decode(Bool.self, forKey: .ir)
        whiteLight = try c.decode(Bool.self, forKey: .whiteLight)
        siren = try c.decode(Bool.self, forKey: .siren)
        mic = try c.decode(Bool.self, forKey: .mic)
        speaker = try c.decode(Bool.self, forKey: .speaker)
        doorbell = try c.decode(Bool.self, forKey: .doorbell)
        aiOnCamera = try c.decode(Bool.self, forKey: .aiOnCamera)
        backchannel = try c.decodeIfPresent(Bool.self, forKey: .backchannel) ?? false
        ptz = try c.decodeIfPresent(Bool.self, forKey: .ptz) ?? false
        nightVision = try c.decodeIfPresent(Bool.self, forKey: .nightVision) ?? false
    }
}

/// A per-camera exempt (privacy / ignore) detection zone — a normalized-coord
/// polygon (each point is [x, y] in 0…1). Mirrors the web `ExemptZone`.
/// Anything whose feet land inside a zone is skipped for detection.
struct ExemptZone: Codable, Sendable, Hashable, Identifiable {
    var name: String
    /// Polygon vertices as [x, y] pairs, normalized to 0…1.
    var points: [[Double]]

    // Stable-ish identity for SwiftUI lists (name + vertex count + first point).
    var id: String {
        let head = points.first.map { "\($0.first ?? 0),\($0.last ?? 0)" } ?? "0"
        return "\(name)|\(points.count)|\(head)"
    }
}

// Hashable so navigation APIs (navigationDestination(item:)) can route on it.
struct Camera: Decodable, Identifiable, Sendable, Hashable {
    struct Toggle: Decodable, Sendable, Hashable {
        let enabled: Bool
    }

    let name: String
    let friendlyName: String
    let model: String
    let ip: String
    let online: Bool
    let source: String
    let needsCredentials: Bool
    let capabilities: CameraCapabilities
    /// Stored tracked-object list, verbatim. `[]` == record-only.
    let detectObjects: [String]
    /// Stored exempt (privacy/ignore) detection zones. Absent on a backend that
    /// predates the feature — decode a missing key as `nil` (treated as none).
    let exemptZones: [ExemptZone]?
    let detect: Toggle
    let record: Toggle
    let detectFps: Int
    let mainUrl: String
    let subUrl: String
    /// Effective server-detection gating (backend applies the global default
    /// when the camera's own value is unset). Absent on a backend that
    /// predates the feature — treat a missing value as `.always`.
    let detectMode: DetectMode?
    /// Raw stored per-camera mode; nil == inherit the global default. Present
    /// only when the backend reports it.
    let detectModeStored: DetectMode?
    /// Live indicator: the camera's own on-camera AI is firing right now.
    /// Optional — absent when the backend doesn't expose the live state.
    let aiActive: Bool?
    /// Live-view audio codec preference ("g711a" | "aac"). "g711a" (the default)
    /// forces the camera's audio encoder to WebRTC-legal G.711A, so live-view
    /// audio works; "aac" gives higher recording quality but drops live-view
    /// audio (go2rtc/WebRTC can't carry AAC). Absent on a backend that predates
    /// the feature — decode a missing key as nil and treat it as "g711a".
    let audioCodec: String?
    /// Per-camera "Smart spotlight": when true, the backend turns this camera's
    /// white-light spotlight on while a person is detected at night, holding it
    /// on until 60s after the last person detection. Only meaningful for
    /// `capabilities.whiteLight` cameras. Absent on a backend that predates the
    /// feature — decode a missing key as nil and treat it as false.
    let smartSpotlight: Bool?
    /// Per-camera "Spotlight hold" (seconds): how long the smart spotlight stays
    /// on AFTER the last person detection at night, before the backend turns it
    /// off. Only meaningful alongside `smartSpotlight` on a `whiteLight` camera.
    /// Valid range 5…600. Absent on a backend that predates the feature — decode
    /// a missing key as nil and treat it as 60.
    let spotlightHoldSeconds: Int?
    /// Software Privacy Mode: the camera is currently a capture kill-switch
    /// target, so the backend serves NO stream, snapshot, recording, detection
    /// or events for it. Backticked because `private` is a Swift keyword — the
    /// JSON key really is "private" (one word, so .convertFromSnakeCase leaves
    /// it alone). Read it through `isPrivate` instead of touching this directly.
    ///
    /// This is the RESOLVED effect, not the configuration: it says "not being
    /// captured" without revealing which cameras/groups an admin selected. It
    /// is the ONLY privacy signal a viewer gets — GET /api/privacy is
    /// admin-only. Absent on a backend that predates the feature — decode a
    /// missing key as nil and treat it as not private.
    let `private`: Bool?

    var id: String { name }

    /// Effective detection mode with the safe default applied.
    var effectiveDetectMode: DetectMode { detectMode ?? .always }

    /// Effective live-view audio codec with the safe default applied.
    var effectiveAudioCodec: String { audioCodec ?? "g711a" }

    /// Effective smart-spotlight flag with the safe default applied.
    var effectiveSmartSpotlight: Bool { smartSpotlight ?? false }

    /// Effective spotlight-hold seconds with the safe default applied.
    var effectiveSpotlightHoldSeconds: Int { spotlightHoldSeconds ?? 60 }

    /// Whether the camera's on-camera AI is currently firing.
    var isAIActive: Bool { aiActive ?? false }

    /// Whether Software Privacy Mode is stopping all capture for this camera.
    /// Fails SAFE on an older backend that omits the key (nil -> false): the
    /// server enforces privacy regardless, so the worst case is a tile that
    /// shows a connection failure instead of the explanatory overlay — never a
    /// tile that leaks video.
    var isPrivate: Bool { `private` ?? false }
}

/// Body for PUT /api/cameras/{name} (CameraUpdate). Identity fields are always
/// sent; a blank username/password keeps the stored device credentials. Every
/// optional below is omitted when nil (synthesized `encodeIfPresent`), which the
/// backend treats as "keep the stored value" — so each editor sends only the
/// fields it changed. The APIClient encoder converts these to snake_case.
struct CameraUpdatePayload: Encodable, Sendable {
    let name: String
    let friendlyName: String
    let model: String
    let ip: String
    var username: String = ""
    var password: String = ""
    var detectObjects: [String]?
    var exemptZones: [ExemptZone]?
    var detectFps: Int?
    /// nil = keep stored; "" = inherit global default; a valid mode = set.
    var detectMode: String?
    var mainUrl: String?
    var subUrl: String?
    /// Live-view audio codec ("g711a" | "aac"); nil = keep stored. When it
    /// changes the backend re-provisions the camera's audio encoder.
    var audioCodec: String?
    /// Per-camera "Smart spotlight" flag; nil = keep stored. Persist-only — the
    /// backend controller reads the stored flag live, so no device call fires on
    /// the PUT.
    var smartSpotlight: Bool?
    /// Per-camera "Spotlight hold" (seconds); nil = keep stored. Valid range
    /// 5…600 (the API validates; the controller clamps defensively). Persist-only
    /// — the backend controller reads the stored value live, so no device call
    /// fires on the PUT.
    var spotlightHoldSeconds: Int?
}

/// Body for POST /api/cameras (CameraInput). Everything the backend needs to
/// adopt a camera: identity, address and the device's own credentials (required
/// here, unlike the update contract where blank means "keep stored"). The two
/// optionals are omitted when nil (synthesized `encodeIfPresent`), which the
/// backend reads as "use the defaults" — `detect_objects` falls back to
/// DEFAULT_DETECT_OBJECTS and `detect_fps` to DEFAULT_DETECT_FPS. The APIClient
/// encoder converts these to snake_case. Admin-only.
struct CameraCreatePayload: Encodable, Sendable {
    let name: String
    let friendlyName: String
    let model: String
    let ip: String
    let username: String
    let password: String
    var detectObjects: [String]?
    var detectFps: Int?
}

/// GET /api/detection/labels response — the active model's selectable class
/// vocabulary for the per-camera object picker.
struct LabelsResponse: Decodable, Sendable {
    let model: String?
    let vocabulary: String?
    let count: Int?
    let labels: [String]
}


// MARK: - Events

struct Event: Decodable, Identifiable, Sendable, Equatable {
    let id: Int
    let frigateId: String
    let camera: String
    /// Primary label (kept for back-compat / sorting / the accent color).
    let label: String
    /// ALL distinct detected classes on this event (multi-object contract).
    /// Absent on a backend that predates the feature — decode a missing key as
    /// nil and fall back to `[label]` via `allLabels`.
    let labels: [String]?
    let count: Int
    let score: Double
    let startTime: Double
    let endTime: Double?
    let hasClip: Bool
    let hasSnapshot: Bool
    let zones: [String]

    /// Every detected class to show, de-duplicated and never empty: the `labels`
    /// list when present, else just `[label]`. Order preserves the wire order
    /// but guarantees the primary `label` is first.
    var allLabels: [String] {
        Event.resolveLabels(primary: label, labels: labels)
    }

    /// Shared label-resolution so Event + EventDetail behave identically.
    static func resolveLabels(primary: String, labels: [String]?) -> [String] {
        var ordered: [String] = []
        var seen = Set<String>()
        for name in ([primary] + (labels ?? [])) where !name.isEmpty {
            if seen.insert(name).inserted { ordered.append(name) }
        }
        return ordered.isEmpty ? [primary] : ordered
    }
}

enum ClipState: String, Decodable, Sendable {
    case ready
    case processing
    case recordingDisabled = "recording_disabled"
    case unavailable
}

struct EventDetail: Decodable, Identifiable, Sendable {
    let id: Int
    let frigateId: String
    let camera: String
    let label: String
    /// ALL distinct detected classes (multi-object contract); nil on an older
    /// backend — fall back to `[label]` via `allLabels`.
    let labels: [String]?
    let count: Int
    let score: Double
    let startTime: Double
    let endTime: Double?
    let hasClip: Bool
    let hasSnapshot: Bool
    let zones: [String]
    let clipUrl: String
    let snapshotUrl: String
    let recordEnabled: Bool
    let clipState: ClipState

    /// Every detected class to show (de-duplicated, primary first, never empty).
    var allLabels: [String] {
        Event.resolveLabels(primary: label, labels: labels)
    }
}

struct EventsPage: Decodable, Sendable {
    let events: [Event]
    let total: Int
}

// MARK: - Suppressions (reject-to-suppress)

/// A learned false-detection suppression: rejecting an event marks it a false
/// detection, learns this suppression, and deletes the event. Listed under
/// Settings › Excluded objects. The decoder uses `.convertFromSnakeCase`, so
/// `createdAt` maps `created_at`; extra JSON fields (foot_x/foot_y/has_thumb/
/// thumbnail_url) are ignored.
struct Suppression: Codable, Identifiable, Sendable {
    let id: Int
    let camera: String
    let label: String
    let createdAt: Double
}

// MARK: - Recordings

struct RecordingCamera: Decodable, Identifiable, Sendable {
    let camera: String
    let friendlyName: String
    let hasRecordings: Bool
    let earliest: Double?
    let latest: Double?

    var id: String { camera }
}

struct RecordingSegment: Decodable, Sendable {
    let start: Double
    let duration: Double
}

struct RecordingRange: Decodable, Sendable {
    let start: Double
    let end: Double
}

struct RecordingIndex: Decodable, Sendable {
    let date: String
    let tzOffset: String
    let segments: [RecordingSegment]
    let ranges: [RecordingRange]

    private enum CodingKeys: String, CodingKey {
        case date, tzOffset, segments, ranges
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        date = try c.decode(String.self, forKey: .date)
        // Backend may emit tz_offset as a string ("-04:00") or a number.
        if let s = try? c.decode(String.self, forKey: .tzOffset) {
            tzOffset = s
        } else if let n = try? c.decode(Double.self, forKey: .tzOffset) {
            tzOffset = String(n)
        } else {
            tzOffset = ""
        }
        segments = try c.decode([RecordingSegment].self, forKey: .segments)
        ranges = try c.decode([RecordingRange].self, forKey: .ranges)
    }
}

// MARK: - Groups

struct CameraGroup: Codable, Identifiable, Sendable {
    let id: Int
    var name: String
    var cameras: [String]
    var position: Int
}

// MARK: - Detection backend + Edge TPU models

/// settings.detection.backend — which silicon runs inference.
enum DetectionBackend: String, Codable, Sendable, CaseIterable {
    /// Use a Coral Edge TPU when one is fitted, else the GPU. The default —
    /// hardware should be picked up without anyone hunting for a setting.
    case auto
    case gpu
    case coral

    var label: String {
        switch self {
        case .auto: return "Automatic"
        case .gpu: return "GPU"
        case .coral: return "Coral Edge TPU"
        }
    }
    var blurb: String {
        switch self {
        case .auto:
            return "Uses a Coral Edge TPU when one is fitted, otherwise the GPU. "
                 + "Fit or remove a Coral and it is picked up on the next restart."
        case .gpu:
            return "D-FINE on CUDA — highest accuracy."
        case .coral:
            return "SSDLite MobileDet on the Edge TPU — about 2 W instead of the GPU."
        }
    }
}

/// The six Edge TPU detectors, mirroring backend `CORAL_MODELS`
/// (app/native/coral.py) and the web list. mAP and latency are Coral's
/// published Edge TPU figures EXCEPT ssdlite_mobiledet, whose 9.6 ms was
/// measured on real hardware.
struct CoralModelInfo: Identifiable, Sendable {
    let key: String
    let label: String
    let map: Double
    let latencyMs: Double
    let note: String
    /// Input square, read off the artifact — never inferred from the filename.
    let inputSize: Int
    /// One-line description, so an Edge TPU model row carries the same weight
    /// of information as a GPU model row rather than reading as a lesser option.
    let blurb: String
    /// Cannot sustain ~10 inferences/sec — about what two cameras at 5 fps
    /// already demand, so frames get dropped.
    let slow: Bool

    var id: String { key }

    static let defaultKey = "ssdlite_mobiledet"

    /// All Edge TPU models are COCO-80 after the backend's sparse COCO-90 remap
    /// — the same vocabulary the GPU models use.
    static let vocabulary = "COCO · 80 classes"

    static let all: [CoralModelInfo] = [
        .init(key: "ssd_mobilenet_v2", label: "SSD MobileNet V2", map: 22.4,
              latencyMs: 7.6, note: "fastest, lowest accuracy", inputSize: 300,
              blurb: "The fastest option. Lowest accuracy — misses small or partly "
                   + "hidden objects.", slow: false),
        .init(key: "ssdlite_mobiledet", label: "SSDLite MobileDet", map: 32.9,
              latencyMs: 9.6, note: "best balance — recommended", inputSize: 320,
              blurb: "Near the accuracy of models 4x slower, at almost the speed of "
                   + "the fastest. The right default for a multi-camera box.", slow: false),
        .init(key: "efficientdet_lite0", label: "EfficientDet-Lite0", map: 25.7,
              latencyMs: 37.4, note: "", inputSize: 320,
              blurb: "A different architecture at the same input size. Slower than "
                   + "MobileDet for less accuracy — mainly useful for comparison.",
              slow: false),
        .init(key: "efficientdet_lite1", label: "EfficientDet-Lite1", map: 30.6,
              latencyMs: 56.3, note: "", inputSize: 384,
              blurb: "A larger input helps distant objects, at roughly 6x MobileDet's "
                   + "inference time.", slow: false),
        .init(key: "efficientdet_lite2", label: "EfficientDet-Lite2", map: 34.0,
              latencyMs: 104.6, note: "may not keep up", inputSize: 448,
              blurb: "More accurate, but over 100 ms per frame — under 10 "
                   + "inferences/sec total.", slow: true),
        .init(key: "efficientdet_lite3", label: "EfficientDet-Lite3", map: 37.7,
              latencyMs: 107.6, note: "highest accuracy, slowest", inputSize: 512,
              blurb: "The most accurate Edge TPU model offered, and the slowest.",
              slow: true),
    ]

    static func find(_ key: String) -> CoralModelInfo {
        all.first { $0.key == key } ?? all.first { $0.key == defaultKey }!
    }
}

// MARK: - Software Privacy Mode (admin-only)

/// GET/POST /api/privacy — the per-camera / per-group capture kill switch.
///
/// **ADMIN-ONLY on both verbs; a viewer gets 403.** Only fetch this from an
/// admin-gated surface. A viewer needs none of it: the dashboard renders its
/// Privacy Mode overlay from `Camera.isPrivate`, which is any-authenticated.
///
/// `cameras`/`groups` are what an admin selected; `privateCameras` is the
/// RESOLVED effective set (direct ∪ every member of a selected group) — that is
/// what the backend gates actually enforce, so the UI must reflect IT rather
/// than inventing its own answer from the selection.
struct PrivacyModeState: Decodable, Sendable {
    let cameras: [String]
    let groups: [Int]
    let privateCameras: [String]
    let enabled: Bool
}

// MARK: - Users (admin-managed)

/// A DB-backed user row from GET /api/users (docs/CONTRACTS.md RBAC addendum).
/// Password hashes are never part of the public shape.
///
/// The built-in admin (username "admin") is env-controlled (ADMIN_PASSWORD) and
/// has NO DB row, so it is never returned here and can't be created, demoted,
/// renamed or deleted through this API — the list is only the *additional*
/// accounts.
struct ManagedUser: Decodable, Identifiable, Sendable {
    let id: Int
    let username: String
    /// Mutable so a role change can be applied optimistically and reverted.
    var role: Role
    /// Unix epoch seconds (the column is a SQLite REAL).
    let createdAt: Double
}

// MARK: - Settings (read a subset; write ONLY a partial patch)
//
// ⚠️ PUT /api/settings is a FULL-DOCUMENT REPLACE where every field carries a
// pydantic default, so a key you omit is RESET — not left alone. A PUT missing
// `notifications.apns.direct.p8` destroys the APNs signing key and silently
// breaks push (verified empirically). iOS therefore NEVER PUTs: it GETs the
// minimal `SettingsDocument` below and writes with `SettingsPatch` through
// PATCH /api/settings, which deep-merges. What we don't model, we can't clobber.

/// GET /api/settings — decodes ONLY the blocks the iOS settings screens read.
/// Every other key (notifications, mqtt, time_sync, and the computed read-only
/// `webrtc` block) is ignored by Decodable, which is exactly the point: this
/// type is deliberately incomplete so it can never be used to round-trip the
/// document back to the server.
struct SettingsDocument: Decodable, Sendable {
    /// `recording` — the retention windows (backend `RecordingSettings`).
    /// Each field is days, 0…365; the per-field fallbacks below are the
    /// backend's own defaults (config.py DEFAULT_SETTINGS) so a backend that
    /// omits a key decodes to what that backend is actually enforcing.
    struct Recording: Decodable, Sendable {
        var continuousDays: Int
        var eventDays: Int
        var snapshotDays: Int

        private enum CodingKeys: String, CodingKey {
            case continuousDays, eventDays, snapshotDays
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            continuousDays = try c.decodeIfPresent(Int.self, forKey: .continuousDays) ?? 7
            eventDays = try c.decodeIfPresent(Int.self, forKey: .eventDays) ?? 14
            snapshotDays = try c.decodeIfPresent(Int.self, forKey: .snapshotDays) ?? 14
        }
    }

    struct Detection: Decodable, Sendable {
        var model: String
        var confidence: Double
        var defaultMode: DetectMode
        /// Which silicon runs inference. Absent on an older backend -> .gpu.
        var backend: DetectionBackend
        /// Edge TPU model key. SEPARATE from `model` (the D-FINE tier) because
        /// the two lists are disjoint — one field would make an invalid
        /// model/backend pair reachable the instant the backend flips.
        var coralModel: String

        private enum CodingKeys: String, CodingKey {
            case model, confidence, defaultMode, backend, coralModel
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            model = try c.decode(String.self, forKey: .model)
            confidence = try c.decode(Double.self, forKey: .confidence)
            // Absent on a backend predating per-camera AI gating — "always" is
            // the backend's own default (continuous server inference).
            defaultMode = try c.decodeIfPresent(DetectMode.self, forKey: .defaultMode) ?? .always
            backend = try c.decodeIfPresent(DetectionBackend.self, forKey: .backend) ?? .gpu
            coralModel = try c.decodeIfPresent(String.self, forKey: .coralModel)
                ?? CoralModelInfo.defaultKey
        }
    }

    struct System: Decodable, Sendable {
        /// Nightly self-restart at a fixed local time. Absent on an older
        /// backend -> off.
        struct AutoRestart: Decodable, Sendable {
            var enabled: Bool
            var time: String  // "HH:MM"
            private enum CodingKeys: String, CodingKey { case enabled, time }
            init(from decoder: Decoder) throws {
                let c = try decoder.container(keyedBy: CodingKeys.self)
                enabled = try c.decodeIfPresent(Bool.self, forKey: .enabled) ?? false
                time = try c.decodeIfPresent(String.self, forKey: .time) ?? "04:00"
            }
            init() { enabled = false; time = "04:00" }
        }

        var publicUrl: String
        var webrtcCandidates: [String]
        var autoRestart: AutoRestart

        private enum CodingKeys: String, CodingKey { case publicUrl, webrtcCandidates, autoRestart }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            publicUrl = try c.decodeIfPresent(String.self, forKey: .publicUrl) ?? ""
            webrtcCandidates = try c.decodeIfPresent([String].self, forKey: .webrtcCandidates) ?? []
            autoRestart = try c.decodeIfPresent(AutoRestart.self, forKey: .autoRestart) ?? AutoRestart()
        }
    }

    /// Only the blocks the app's notification screens read. Decoding just what
    /// we render keeps this type unable to round-trip the document, which is
    /// the whole invariant above.
    ///
    /// `apns` USED to be excluded on the grounds that its `direct.p8` was
    /// Apple's signing key and a private key has no business on a phone. That
    /// reason is gone with `direct` — the block is now a mode plus a relay URL,
    /// no secret in it — so the app configures APNs too. Do not re-add a `p8`
    /// here if `direct` ever returns.
    struct Notifications: Decodable, Sendable {
        /// `notifications.apns` — delivery to THIS app: a native notification
        /// and the CallKit doorbell ring. Optional-with-fallback like `ntfy`
        /// below: absent on an older backend, and `mode` may arrive as the
        /// retired `"direct"` from one that hasn't migrated yet, which decodes
        /// to `.off` rather than throwing and blanking the whole screen.
        struct Apns: Decodable, Sendable {
            enum Mode: String, Decodable, Sendable {
                case relay, off
            }

            var mode: Mode
            /// `relayUrl`, NOT `relayURL`: the decoder's `.convertFromSnakeCase`
            /// rewrites the wire's `relay_url` to `relayUrl` BEFORE matching
            /// CodingKeys, so a `URL`-cased property (or a hand-written
            /// `= "relay_url"` key) silently never matches and decodes to "".
            var relayUrl: String

            private enum CodingKeys: String, CodingKey { case mode, relayUrl }

            init(from decoder: Decoder) throws {
                let c = try decoder.container(keyedBy: CodingKeys.self)
                // Decode the raw string, not the enum: a legacy "direct" would
                // make a plain `decodeIfPresent(Mode.self)` THROW, and this
                // struct's failure takes the whole settings document with it.
                let raw = try c.decodeIfPresent(String.self, forKey: .mode) ?? "off"
                mode = Mode(rawValue: raw) ?? .off
                relayUrl = try c.decodeIfPresent(String.self, forKey: .relayUrl) ?? ""
            }

            init() {
                mode = .off
                relayUrl = ""
            }
        }

        /// `notifications.ntfy` — push with no Apple developer account.
        /// Every field is optional-with-fallback: the block is absent on a
        /// backend that predates it, AND on one old enough to have stripped it
        /// as a legacy block (ntfy support was removed once, then restored).
        struct Ntfy: Decodable, Sendable {
            var enabled: Bool
            var server: String
            var topic: String
            var authToken: String
            var priority: Int
            var attachSnapshot: Bool

            private enum CodingKeys: String, CodingKey {
                case enabled, server, topic, authToken, priority, attachSnapshot
            }

            init(from decoder: Decoder) throws {
                let c = try decoder.container(keyedBy: CodingKeys.self)
                enabled = try c.decodeIfPresent(Bool.self, forKey: .enabled) ?? false
                server = try c.decodeIfPresent(String.self, forKey: .server) ?? "https://ntfy.sh"
                // No default topic, on purpose: the topic is a shared secret,
                // so the UI generates an unguessable one (see PhonePushSettingsView).
                topic = try c.decodeIfPresent(String.self, forKey: .topic) ?? ""
                authToken = try c.decodeIfPresent(String.self, forKey: .authToken) ?? ""
                priority = try c.decodeIfPresent(Int.self, forKey: .priority) ?? 4
                attachSnapshot = try c.decodeIfPresent(Bool.self, forKey: .attachSnapshot) ?? true
            }

            init() {
                enabled = false
                server = "https://ntfy.sh"
                topic = ""
                authToken = ""
                priority = 4
                attachSnapshot = true
            }
        }

        var apns: Apns
        var ntfy: Ntfy
        /// `notifications.camera_down_alerts` — push me when a camera goes
        /// offline. Absent on an older backend -> off (never surprises with a
        /// storm of alerts the first time this field appears).
        var cameraDownAlerts: Bool
        /// Notification RULES — apply to EVERY channel. Absent on an older
        /// backend -> the backend's own defaults, so the screen still renders.
        var enabled: Bool
        var labels: [String]
        var cooldownSeconds: Int
        var minScore: Double
        var drawBoxes: Bool

        private enum CodingKeys: String, CodingKey {
            case apns, ntfy, cameraDownAlerts, enabled, labels, cooldownSeconds, minScore, drawBoxes
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            apns = try c.decodeIfPresent(Apns.self, forKey: .apns) ?? Apns()
            ntfy = try c.decodeIfPresent(Ntfy.self, forKey: .ntfy) ?? Ntfy()
            cameraDownAlerts = try c.decodeIfPresent(Bool.self, forKey: .cameraDownAlerts) ?? false
            enabled = try c.decodeIfPresent(Bool.self, forKey: .enabled) ?? true
            labels = try c.decodeIfPresent([String].self, forKey: .labels)
                ?? ["person", "dog", "cat", "car"]
            cooldownSeconds = try c.decodeIfPresent(Int.self, forKey: .cooldownSeconds) ?? 60
            minScore = try c.decodeIfPresent(Double.self, forKey: .minScore) ?? 0.7
            drawBoxes = try c.decodeIfPresent(Bool.self, forKey: .drawBoxes) ?? true
        }

        init() {
            apns = Apns()
            ntfy = Ntfy()
            cameraDownAlerts = false
            enabled = true
            labels = ["person", "dog", "cat", "car"]
            cooldownSeconds = 60
            minScore = 0.7
            drawBoxes = true
        }
    }

    /// Home Assistant MQTT integration. Absent on an older backend -> disabled
    /// defaults. `password` arrives MASKED from the server (GET masks secrets),
    /// so a blank/masked value on save means "keep the stored password".
    struct Mqtt: Decodable, Sendable {
        var enabled: Bool
        var host: String
        var port: Int
        var username: String
        var password: String
        var discoveryPrefix: String
        var baseTopic: String
        private enum CodingKeys: String, CodingKey {
            case enabled, host, port, username, password, discoveryPrefix, baseTopic
        }
        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            enabled = try c.decodeIfPresent(Bool.self, forKey: .enabled) ?? false
            host = try c.decodeIfPresent(String.self, forKey: .host) ?? ""
            port = try c.decodeIfPresent(Int.self, forKey: .port) ?? 1883
            username = try c.decodeIfPresent(String.self, forKey: .username) ?? ""
            password = try c.decodeIfPresent(String.self, forKey: .password) ?? ""
            discoveryPrefix = try c.decodeIfPresent(String.self, forKey: .discoveryPrefix) ?? "homeassistant"
            baseTopic = try c.decodeIfPresent(String.self, forKey: .baseTopic) ?? "vigilume"
        }
        init() {
            enabled = false; host = ""; port = 1883; username = ""; password = ""
            discoveryPrefix = "homeassistant"; baseTopic = "vigilume"
        }
    }

    /// Automatic camera time provisioning. Absent on an older backend -> on with
    /// the server's default timezone (empty here; the view shows the real value).
    struct TimeSync: Decodable, Sendable {
        var autoSync: Bool
        var timezone: String
        private enum CodingKeys: String, CodingKey { case autoSync, timezone }
        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            autoSync = try c.decodeIfPresent(Bool.self, forKey: .autoSync) ?? true
            timezone = try c.decodeIfPresent(String.self, forKey: .timezone) ?? ""
        }
        init() { autoSync = true; timezone = "" }
    }

    var recording: Recording
    var detection: Detection
    var system: System
    /// Absent on an older backend -> defaults, so the screen still renders.
    var notifications: Notifications
    var mqtt: Mqtt
    var timeSync: TimeSync

    private enum CodingKeys: String, CodingKey {
        case recording, detection, system, notifications, mqtt, timeSync
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        recording = try c.decode(Recording.self, forKey: .recording)
        detection = try c.decode(Detection.self, forKey: .detection)
        system = try c.decode(System.self, forKey: .system)
        notifications = try c.decodeIfPresent(Notifications.self, forKey: .notifications)
            ?? Notifications()
        mqtt = try c.decodeIfPresent(Mqtt.self, forKey: .mqtt) ?? Mqtt()
        timeSync = try c.decodeIfPresent(TimeSync.self, forKey: .timeSync) ?? TimeSync()
    }
}

/// PATCH /api/settings body — a PARTIAL document the backend deep-merges over
/// the stored one (validation + side-effects identical to PUT; 422 on invalid).
/// Every sub-object is OPTIONAL and the synthesized `Encodable` emits them with
/// `encodeIfPresent`, so a nil subtree is omitted from the JSON entirely and the
/// server leaves it untouched. Send only the subtree you actually edited.
///
/// ⚠️ Do NOT hand-roll `encode(to:)` here, and do NOT make a subtree
/// non-optional. The synthesized encoder's `encodeIfPresent` is load-bearing: a
/// nil subtree written as an explicit JSON `null` would reach the backend's
/// `_deep_merge` and overwrite that whole block with None — the same
/// data-destroying shape as the PUT this type exists to avoid.
struct SettingsPatch: Encodable, Sendable {
    /// Retention windows. Each is days and the backend validates `ge=0, le=365`
    /// — keep senders inside 0…365 so the UI can't produce a 422.
    struct Recording: Encodable, Sendable {
        var continuousDays: Int
        var eventDays: Int
        var snapshotDays: Int
    }

    struct Detection: Encodable, Sendable {
        var model: String
        var confidence: Double
        var defaultMode: DetectMode
        /// OPTIONAL on purpose: JSONEncoder omits nil, and the backend
        /// deep-merges, so a screen that does not touch the backend/model
        /// choice cannot clobber it. Never give these non-nil defaults.
        var backend: DetectionBackend?
        var coralModel: String?
    }

    struct System: Encodable, Sendable {
        /// Nightly self-restart. `time` is "HH:MM" (backend max 5 chars).
        struct AutoRestart: Encodable, Sendable {
            var enabled: Bool
            var time: String
        }
        var publicUrl: String
        var webrtcCandidates: [String]
        /// Optional (nil omitted) so a screen editing only auto-restart leaves
        /// the public URL + WebRTC candidates untouched.
        var autoRestart: AutoRestart?
    }

    /// `apns` and `ntfy`, each optional. Every property here is Optional and
    /// `JSONEncoder` omits nil, so a patch carrying only `ntfy` leaves `apns`
    /// untouched — the backend's `_deep_merge` only touches keys present in the
    /// body. That is what makes it safe for two screens to write the same
    /// block. NEVER give these non-nil defaults.
    struct Notifications: Encodable, Sendable {
        /// `mode` is a backend Literal of exactly `relay | off` — the retired
        /// `direct` would 422. `relayURL` is required when mode is `relay`.
        struct Apns: Encodable, Sendable {
            var mode: String
            /// `relayUrl` — `.convertToSnakeCase` emits `relay_url`. See the
            /// decode-side note; the two must agree or a save is a silent no-op.
            var relayUrl: String
        }

        /// The backend validates: `server` http(s), `topic`
        /// `^[A-Za-z0-9_-]{1,64}$`, `priority` 1…5 — keep senders inside those
        /// so the UI can't produce a 422.
        struct Ntfy: Encodable, Sendable {
            var enabled: Bool
            var server: String
            var topic: String
            var authToken: String
            var priority: Int
            var attachSnapshot: Bool
        }

        var apns: Apns?
        var ntfy: Ntfy?
        /// `camera_down_alerts` — nil omitted, so a screen flipping only this
        /// leaves the push channels untouched.
        var cameraDownAlerts: Bool?
        /// Notification RULES — each nil-omitted, so a screen can flip one
        /// without clobbering the channels or the other rules. Backend bounds:
        /// cooldownSeconds 0…86400, minScore 0…1, labels each ≤32 chars lowercased.
        var enabled: Bool?
        var labels: [String]?
        var cooldownSeconds: Int?
        var minScore: Double?
        var drawBoxes: Bool?
    }

    /// Home Assistant MQTT. Every field Optional (nil omitted) — send only what
    /// changed. Leave `password` nil to keep the stored secret; send a new
    /// non-empty value to replace it (the ntfy-token carry-forward pattern).
    struct Mqtt: Encodable, Sendable {
        var enabled: Bool?
        var host: String?
        var port: Int?
        var username: String?
        var password: String?
        var discoveryPrefix: String?
        var baseTopic: String?
    }

    /// Automatic camera time provisioning. Optional fields, nil-omitted.
    struct TimeSync: Encodable, Sendable {
        var autoSync: Bool?
        var timezone: String?
    }

    var recording: Recording?
    var detection: Detection?
    var system: System?
    var notifications: Notifications?
    var mqtt: Mqtt?
    var timeSync: TimeSync?
}

/// One registered APNs device (GET /api/notifications/apns/devices) — only an
/// 8-char token prefix, never the full token. Lets a phone confirm it registered.
struct ApnsDevice: Decodable, Identifiable, Sendable {
    let deviceTokenPrefix: String
    let deviceName: String
    let createdAt: Double
    var id: String { deviceTokenPrefix }
}

// MARK: - Detection models (read-only in the app)

struct DetectionModel: Decodable, Identifiable, Sendable {
    let key: String
    let tier: String
    let label: String
    let blurb: String
    let sizeBytes: Int
    let inputSize: Int
    let approxMap: Double
    let mapDataset: String
    let recommendedFor: String
    let vocabulary: String
    let numClasses: Int
    let state: String        // absent | downloading | verifying | ready | error
    let progressPct: Double
    let active: Bool
    let loaded: Bool
    let shaOk: Bool?
    let detail: String?

    var id: String { key }
}

struct DetectionModelsResponse: Decodable, Sendable {
    let active: String
    let device: String?
    let models: [DetectionModel]
}

// MARK: - System health

struct SystemHealth: Decodable, Sendable {
    struct Detector: Decodable, Sendable {
        let kind: String
        let ready: Bool
        let device: String?
        let model: String?
    }

    let status: String
    let version: String
    let detector: Detector
    let go2rtc: Bool
    let camerasOnline: Int
}

// MARK: - Detector self-test (GET /api/system/detector, admin)

/// Detector self-test + per-camera ingest health. Core fields required; the
/// rest optional to tolerate an older backend that omits them.
struct DetectorStatus: Decodable, Sendable {
    let ready: Bool
    let device: String
    let model: String
    let modelShaOk: Bool?
    let lastInferenceMs: Double?
    let consecutiveFailures: Int?
    let needsReinit: Bool?
    let lastReinitAgeS: Double?
    let modelState: String?
    let modelProgressPct: Int?
    let perCamera: [Camera]

    struct Camera: Decodable, Identifiable, Sendable {
        let name: String
        let ingestOk: Bool?
        let fps: Double?
        let lastFrameAgeS: Double?
        let stalled: Bool?
        let respawns: Int?
        let aiActive: Bool?
        var id: String { name }
    }
}

// MARK: - Camera health (GET /api/system/camera-health, require_auth)

/// Per-camera RTSP-port reachability over a window. Uptime is CONNECTIVITY, not
/// a guarantee footage recorded (the UI copy must say so).
struct CameraHealthReport: Decodable, Sendable {
    struct Window: Decodable, Sendable {
        let since: Double
        let until: Double
        let hours: Double
    }
    struct Row: Decodable, Identifiable, Sendable {
        let camera: String
        let uptimePct: Double?
        let online: Bool?
        let downCount: Int
        let downSeconds: Double
        let downs: [Down]
        var id: String { camera }
        struct Down: Decodable, Identifiable, Sendable {
            let start: Double
            let end: Double
            let seconds: Double
            var id: String { "\(start)-\(end)" }
        }
    }
    let window: Window
    let cameras: [Row]
}

/// POST /api/integrations/mqtt/test result — a draft-config probe (no save).
struct MqttTestResult: Decodable, Sendable {
    let ok: Bool
    let detail: String?
}

// MARK: - WebSocket frames (/api/ws)

/// One decoded frame off `WS /api/ws?token=`.
enum WSMessage: Sendable {
    case eventNew(Event)
    case eventUpdate(Event)
    case eventEnd(Event)
    case doorbell(Event)
    /// Online/offline updates keyed by camera name (one frame may carry many).
    case cameraStatus([String: Bool])
    /// The camera LIST or its per-camera flags changed — refetch GET /api/cameras.
    ///
    /// Load-bearing for Privacy Mode. Toggling privacy does not change a
    /// camera's online status (it is a software gate; the camera's own RTSP
    /// port is untouched), so `cameraStatus` says nothing about it. The only
    /// signal is the `private` flag on the camera row, and without reacting to
    /// this message the app keeps a stale copy — showing live tiles for a camera
    /// the backend has stopped capturing, instead of the Privacy Mode overlay.
    case camerasChanged
    case modelStatus(key: String, state: String, progressPct: Double, active: Bool, loaded: Bool)
    case unknown(type: String)

    /// Decode a raw text frame. Unknown/garbled frames come back as
    /// `.unknown` — never throws for forward compatibility.
    static func decode(_ text: String) -> WSMessage {
        guard let data = text.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = obj["type"] as? String
        else { return .unknown(type: "") }

        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase

        func event() -> Event? {
            guard let raw = obj["event"],
                  let eventData = try? JSONSerialization.data(withJSONObject: raw)
            else { return nil }
            return try? decoder.decode(Event.self, from: eventData)
        }

        switch type {
        case "event_new":
            if let e = event() { return .eventNew(e) }
        case "event_update":
            if let e = event() { return .eventUpdate(e) }
        case "event_end":
            if let e = event() { return .eventEnd(e) }
        case "doorbell":
            if let e = event() { return .doorbell(e) }
        case "camera_status":
            // Shape is loose in the contract — mirror the web client's
            // parseCameraStatus: {camera|name, online} or a map under
            // {cameras|status} of name -> Bool | {online: Bool}.
            var updates: [String: Bool] = [:]
            if let name = (obj["camera"] as? String) ?? (obj["name"] as? String),
               let online = obj["online"] as? Bool {
                updates[name] = online
            } else if let map = (obj["cameras"] ?? obj["status"]) as? [String: Any] {
                for (name, value) in map {
                    if let online = value as? Bool {
                        updates[name] = online
                    } else if let nested = value as? [String: Any],
                              let online = nested["online"] as? Bool {
                        updates[name] = online
                    }
                }
            }
            if !updates.isEmpty { return .cameraStatus(updates) }
        case "cameras_changed":
            return .camerasChanged
        case "model_status":
            return .modelStatus(
                key: (obj["key"] as? String) ?? "",
                state: (obj["state"] as? String) ?? "",
                progressPct: (obj["progress_pct"] as? Double) ?? 0,
                active: (obj["active"] as? Bool) ?? false,
                loaded: (obj["loaded"] as? Bool) ?? false
            )
        default:
            break
        }
        return .unknown(type: type)
    }
}
