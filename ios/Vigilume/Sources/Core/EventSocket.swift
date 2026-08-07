import Foundation
import Combine

/// WebSocket client for `WS /api/ws?token=` with automatic reconnect +
/// exponential backoff. Publishes decoded `WSMessage` frames.
@MainActor
final class EventSocket: ObservableObject {
    enum ConnectionState: Equatable {
        case disconnected
        case connecting
        case connected
    }

    @Published private(set) var state: ConnectionState = .disconnected

    /// Decoded frames (event_new/update/end, doorbell, camera_status,
    /// model_status). Delivered on the main actor.
    let messages = PassthroughSubject<WSMessage, Never>()

    /// Fires on every successful (re)connect — subscribers refetch lists
    /// to close any gap while disconnected.
    let reconnected = PassthroughSubject<Void, Never>()

    private var task: URLSessionWebSocketTask?
    private var receiveLoop: Task<Void, Never>?
    private var url: URL?
    private var shouldRun = false
    private var protocols: [String] = []
    private var attempt = 0
    private var hasConnectedOnce = false

    /// Start (or restart) the socket against the given URL.
    func connect(url: URL, protocols: [String] = []) {
        disconnect()
        self.url = url
        self.protocols = protocols
        shouldRun = true
        attempt = 0
        hasConnectedOnce = false
        open()
    }

    func disconnect() {
        shouldRun = false
        receiveLoop?.cancel()
        receiveLoop = nil
        task?.cancel(with: .normalClosure, reason: nil)
        task = nil
        state = .disconnected
    }

    private func open() {
        guard shouldRun, let url else { return }
        state = .connecting
        // `protocols` carries the bearer token; the server must echo "bearer"
        // back or the handshake is rejected.
        let wsTask = protocols.isEmpty
            ? URLSession.shared.webSocketTask(with: url)
            : URLSession.shared.webSocketTask(with: url, protocols: protocols)
        task = wsTask
        wsTask.resume()

        // The server only pushes frames when something happens, so don't wait
        // for one to declare the socket live — a pong proves the connection.
        wsTask.sendPing { [weak self] error in
            guard error == nil else { return }
            Task { @MainActor [weak self] in
                self?.markConnected(for: wsTask)
            }
        }

        receiveLoop = Task { [weak self] in
            await self?.runReceiveLoop(on: wsTask)
        }
    }

    private func markConnected(for wsTask: URLSessionWebSocketTask) {
        guard shouldRun, task === wsTask, state != .connected else { return }
        state = .connected
        attempt = 0
        if hasConnectedOnce { reconnected.send() }
        hasConnectedOnce = true
    }

    private func runReceiveLoop(on wsTask: URLSessionWebSocketTask) async {
        while shouldRun, !Task.isCancelled, task === wsTask {
            do {
                let frame = try await wsTask.receive()
                markConnected(for: wsTask)   // first frame also proves liveness
                if case .string(let text) = frame {
                    messages.send(WSMessage.decode(text))
                }
            } catch {
                break
            }
        }
        guard shouldRun, task === wsTask else { return }
        state = .disconnected
        scheduleReconnect()
    }

    private func scheduleReconnect() {
        guard shouldRun else { return }
        attempt += 1
        // 1s, 2s, 4s ... capped at 30s (mirrors the web client's backoff).
        let delay = min(30.0, pow(2.0, Double(attempt - 1)))
        Task { [weak self] in
            try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
            guard let self, self.shouldRun else { return }
            self.open()
        }
    }
}
