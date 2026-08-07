import SwiftUI

/// Shared transport row for the multi-camera Timeline, mounted under the unified
/// scrub bar. Renders NO video of its own — it only reflects and mutates the
/// coordinator's shared playback state (play/pause, ±10 s skip, speed, mute,
/// jump-to-newest), so every synchronized player reacts at once. Mirrors
/// TimelineTransport.tsx.
struct TimelineTransportView: View {
    @ObservedObject var coordinator: TimelineSyncCoordinator
    /// Disabled when nothing is playable (no coverage on the day).
    let disabled: Bool

    private let rates: [Float] = [0.5, 1, 2, 4]

    var body: some View {
        HStack(spacing: 18) {
            Button { coordinator.skip(by: -10) } label: {
                Image(systemName: "gobackward.10").font(.title3)
            }
            Button { coordinator.togglePlay() } label: {
                Image(systemName: coordinator.isPlaying ? "pause.circle.fill" : "play.circle.fill")
                    .font(.system(size: 44))
            }
            Button { coordinator.skip(by: 10) } label: {
                Image(systemName: "goforward.10").font(.title3)
            }

            Spacer()

            Button { coordinator.toggleMute() } label: {
                Image(systemName: coordinator.muted ? "speaker.slash.fill" : "speaker.wave.2.fill")
                    .font(.title3)
            }
            .accessibilityLabel(coordinator.muted ? "Unmute" : "Mute")

            Menu {
                ForEach(rates, id: \.self) { speed in
                    Button {
                        coordinator.setRate(speed)
                    } label: {
                        if coordinator.rate == speed {
                            Label(speedLabel(speed), systemImage: "checkmark")
                        } else {
                            Text(speedLabel(speed))
                        }
                    }
                }
            } label: {
                Text(speedLabel(coordinator.rate))
                    .font(.footnote.weight(.semibold))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(Capsule().fill(Theme.surface))
                    .overlay(Capsule().stroke(Theme.border, lineWidth: 1))
            }

            Button { coordinator.jumpToNewest() } label: {
                Image(systemName: "forward.end").font(.title3)
            }
            .disabled(!coordinator.canJumpToNewest)
        }
        .tint(Theme.accent)
        .disabled(disabled)
        .padding(.horizontal, 20)
    }

    private func speedLabel(_ speed: Float) -> String {
        speed == 0.5 ? "0.5×" : "\(Int(speed))×"
    }
}
