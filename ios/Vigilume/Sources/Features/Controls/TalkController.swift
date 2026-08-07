import AVFAudio
import Foundation
import WebRTC

// Push-to-talk uplink per docs/CONTRACTS.md:
//   WS /api/cameras/{name}/talk?token=   (session JWT only)
//   Client sends BINARY frames of raw little-endian Int16 PCM, mono, 8 kHz.
//   Text frames ignored except {"type":"stop"}. One active talker per camera.
//   Close codes: 4003 no speaker, 4009 busy, 4502 camera rejected audio,
//   1000 clean stop (client stop or the 120 s session cap).

// MARK: - TalkController

/// Owns the mic pipeline + talk WebSocket for one push-to-talk session.
/// UI drives it with start(url:)/stop(); observes `state` and `alertMessage`.
@MainActor
final class TalkController: ObservableObject {
    enum State: Equatable {
        case idle
        case connecting   // permission + audio session + WS setup
        case live         // streaming mic audio
    }

    @Published private(set) var state: State = .idle
    /// Human-readable failure surfaced as an alert (close codes, mic denial…).
    @Published var alertMessage: String?

    private var socketTask: URLSessionWebSocketTask?
    private var streamer: TalkAudioStreamer?
    private var audioSessionActive = false

    /// Begin a talk session. No-op unless idle.
    /// Bearer-token subprotocols for the handshake (APIClient.wsSubprotocols).
    private var protocols: [String] = []

    func start(url: URL, protocols: [String] = []) {
        guard state == .idle else { return }
        state = .connecting
        self.protocols = protocols
        Task { await begin(url: url) }
    }

    /// End the session cleanly: {"type":"stop"} + close 1000, release audio.
    func stop() {
        guard state != .idle else { return }
        teardown(sendStop: true)
        state = .idle
    }

    // MARK: Session lifecycle

    private func begin(url: URL) async {
        let granted = await AVAudioApplication.requestRecordPermission()
        guard granted else {
            state = .idle
            alertMessage = "Microphone access is off. Enable it for Vigilume in Settings > Privacy & Security > Microphone."
            return
        }
        // The user may have released the button while the permission prompt
        // was up (or a second start raced) — only proceed if still connecting.
        guard state == .connecting else { return }

        // Drive the session through WebRTC's RTCAudioSession wrapper (not a bare
        // AVAudioSession) so the live-receive WebRTC audio unit stays in sync:
        // a bare setCategory/​setActive here is invisible to WebRTC and would
        // silence its playout, and a bare deactivate on stop used to leave the
        // session inactive under WebRTC — killing receive audio after a talk.
        // A fresh config (NOT the shared .webRTC() singleton) keeps the mic-less
        // .playback default intact for the viewer path.
        do {
            let rtcSession = RTCAudioSession.sharedInstance()
            rtcSession.lockForConfiguration()
            defer { rtcSession.unlockForConfiguration() }
            let config = RTCAudioSessionConfiguration()
            // No .duckOthers: ducking would silence the app's OWN live receive
            // audio, which we keep audible during talk for a real 2-way call.
            config.category = AVAudioSession.Category.playAndRecord.rawValue
            config.mode = AVAudioSession.Mode.voiceChat.rawValue
            config.categoryOptions = [.defaultToSpeaker]
            try rtcSession.setConfiguration(config, active: true)
            audioSessionActive = true
        } catch {
            state = .idle
            alertMessage = "Couldn't start the microphone: \(error.localizedDescription)"
            return
        }

        // Bearer token via subprotocol — see APIClient.wsSubprotocols.
        let task = protocols.isEmpty
            ? URLSession.shared.webSocketTask(with: url)
            : URLSession.shared.webSocketTask(with: url, protocols: protocols)
        socketTask = task
        task.resume()
        receiveLoop(task)

        // Mic tap fires on the audio render thread; frames go straight to the
        // socket (URLSessionWebSocketTask.send is thread-safe).
        let pipeline = TalkAudioStreamer { [weak task] frame in
            task?.send(.data(frame)) { _ in }
        }
        do {
            try pipeline.start()
        } catch {
            teardown(sendStop: false)
            state = .idle
            alertMessage = "Couldn't capture audio: \(error.localizedDescription)"
            return
        }
        streamer = pipeline
        state = .live
    }

    private func receiveLoop(_ task: URLSessionWebSocketTask) {
        task.receive { [weak self] result in
            Task { @MainActor [weak self] in
                guard let self, self.socketTask === task else { return }
                switch result {
                case .success:
                    self.receiveLoop(task)   // server frames are ignored
                case .failure:
                    // Socket died or the server closed it — surface the code.
                    let code = task.closeCode.rawValue
                    self.teardown(sendStop: false)
                    self.state = .idle
                    self.alertMessage = Self.message(forCloseCode: code)
                }
            }
        }
    }

    private func teardown(sendStop: Bool) {
        streamer?.stop()
        streamer = nil
        if let task = socketTask {
            socketTask = nil
            if sendStop {
                task.send(.string(#"{"type":"stop"}"#)) { _ in }
                task.cancel(with: .normalClosure, reason: nil)
            } else {
                task.cancel(with: .goingAway, reason: nil)
            }
        }
        if audioSessionActive {
            audioSessionActive = false
            // Balance our RTCAudioSession activation. Because RTCAudioSession is
            // reference-counted, this only truly deactivates the underlying
            // session if WebRTC isn't also holding it active — so live receive
            // audio keeps playing through a talk teardown instead of going dead.
            let rtcSession = RTCAudioSession.sharedInstance()
            rtcSession.lockForConfiguration()
            try? rtcSession.setActive(false)
            rtcSession.unlockForConfiguration()
        }
    }

    /// Contract close codes -> user-facing text. nil == no alert needed.
    private static func message(forCloseCode code: Int) -> String? {
        switch code {
        case 4003:
            return "This camera has no speaker — talk isn't supported."
        case 4009:
            return "Someone else is already talking through this camera."
        case 4502:
            return "The camera rejected the audio or is unreachable."
        case 1000:
            return nil   // clean stop (our close or the 120 s session cap)
        default:
            return "Talk connection lost."
        }
    }
}

// MARK: - TalkAudioStreamer

/// Non-isolated mic pipeline: AVAudioEngine input tap -> AVAudioConverter
/// downsample to 8 kHz Int16 mono (little-endian on all iOS hardware) ->
/// `send` callback per converted buffer.
final class TalkAudioStreamer {
    struct CaptureError: LocalizedError {
        let errorDescription: String?
    }

    private let engine = AVAudioEngine()
    private let send: (Data) -> Void
    private var tapInstalled = false

    init(send: @escaping (Data) -> Void) {
        self.send = send
    }

    func start() throws {
        let input = engine.inputNode
        let inFormat = input.outputFormat(forBus: 0)
        guard inFormat.sampleRate > 0, inFormat.channelCount > 0 else {
            throw CaptureError(errorDescription: "No microphone input is available.")
        }
        guard
            let outFormat = AVAudioFormat(
                commonFormat: .pcmFormatInt16,
                sampleRate: 8000,
                channels: 1,
                interleaved: true
            ),
            let converter = AVAudioConverter(from: inFormat, to: outFormat)
        else {
            throw CaptureError(errorDescription: "Audio format conversion is unavailable.")
        }

        let ratio = outFormat.sampleRate / inFormat.sampleRate
        let sendFrame = send
        input.installTap(onBus: 0, bufferSize: 2048, format: inFormat) { buffer, _ in
            let capacity = AVAudioFrameCount((Double(buffer.frameLength) * ratio).rounded(.up)) + 16
            guard let out = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: capacity) else {
                return
            }
            var fed = false
            let status = converter.convert(to: out, error: nil) { _, outStatus in
                if fed {
                    outStatus.pointee = .noDataNow
                    return nil
                }
                fed = true
                outStatus.pointee = .haveData
                return buffer
            }
            guard
                status != .error,
                out.frameLength > 0,
                let samples = out.int16ChannelData
            else { return }
            sendFrame(Data(
                bytes: samples[0],
                count: Int(out.frameLength) * MemoryLayout<Int16>.size
            ))
        }
        tapInstalled = true

        engine.prepare()
        do {
            try engine.start()
        } catch {
            input.removeTap(onBus: 0)
            tapInstalled = false
            throw error
        }
    }

    func stop() {
        if tapInstalled {
            engine.inputNode.removeTap(onBus: 0)
            tapInstalled = false
        }
        engine.stop()
    }
}
