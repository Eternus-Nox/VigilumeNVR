import AVFoundation
import SwiftUI
import UIKit

/// AVPlayerLayer wrapper — fill for grid tiles, fit for the full-screen view.
struct PlayerLayerView: UIViewRepresentable {
    let player: AVPlayer?
    var videoGravity: AVLayerVideoGravity = .resizeAspectFill

    final class PlayerContainerView: UIView {
        override static var layerClass: AnyClass { AVPlayerLayer.self }
        var playerLayer: AVPlayerLayer { layer as! AVPlayerLayer }
    }

    func makeUIView(context: Context) -> PlayerContainerView {
        let view = PlayerContainerView()
        view.backgroundColor = .clear
        view.playerLayer.videoGravity = videoGravity
        return view
    }

    func updateUIView(_ uiView: PlayerContainerView, context: Context) {
        uiView.playerLayer.videoGravity = videoGravity
        if uiView.playerLayer.player !== player {
            uiView.playerLayer.player = player
        }
    }
}
