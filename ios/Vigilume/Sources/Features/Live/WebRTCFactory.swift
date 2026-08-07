import WebRTC

/// One process-wide `RTCPeerConnectionFactory`. Building a factory per
/// connection re-initializes SSL each time and wastes resources, so the
/// recvonly receive path (`WHEPPlayer`) shares this single instance. The
/// decoder factory lets that recvonly video path decode; the encoder factory is
/// retained for parity with the standard factory build.
enum WebRTCFactory {
    static let shared: RTCPeerConnectionFactory = {
        RTCInitializeSSL()
        return RTCPeerConnectionFactory(
            encoderFactory: RTCDefaultVideoEncoderFactory(),
            decoderFactory: RTCDefaultVideoDecoderFactory()
        )
    }()

    /// Build the factory eagerly, OFF the main actor.
    ///
    /// `shared` is a lazy static, so whoever touches it first pays for
    /// `RTCInitializeSSL()` + factory construction (~100-300 ms). Left alone,
    /// that first touch happens inside `WHEPPlayer.connect` — which is
    /// `@MainActor` — while the first camera tile is coming up, so the cost
    /// lands on the main thread and janks the grid layout. Calling this from a
    /// detached task at launch moves it off that critical path.
    ///
    /// Idempotent and thread-safe: Swift initializes a lazy static exactly once
    /// (`swift_once`), so repeated or concurrent calls are free after the
    /// first, and a tile that connects mid-warm simply blocks on the same
    /// one-time initialization it would have run itself.
    static func warm() {
        _ = shared
    }
}
