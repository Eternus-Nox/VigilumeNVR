import SwiftUI

/// PTZ pad + preset row for a caps.ptz camera (the IP3M-941B dome). Pure UI:
/// it owns no networking — tap-to-step, the speed slider and preset gestures
/// bubble up through closures so `CameraDetailView` runs them against
/// `APIClient.ptz(...)` and keeps the in-flight / saved-slot / error state.
///
/// Movement is tap-to-step: each tap on a direction fires `onStep(direction)`,
/// which sends ONE small step (no hold, no separate stop) — this replaces the
/// old press-and-hold that ran away. The 1–8 speed slider sets the step
/// magnitude (parity with the web pad).
///
/// Presets are a HOLD-button model (the fix):
///   • TAP    = recall (preset_goto) — only when that slot is saved.
///   • HOLD   = save the current position (preset_set) — the affordance that
///              used to be missing/unreliable.
///   • ✕      = clear the slot (preset_clear).
/// Saved vs empty is shown explicitly, and each fired action flashes a caption.
struct PTZControlsView: View {
    let onStep: (PTZDirection) -> Void
    let onPresetGoto: (Int) -> Void
    let onPresetSet: (Int) -> Void
    let onPresetClear: (Int) -> Void
    /// Which slots (1…3) currently hold a saved position.
    let savedPresets: Set<Int>
    /// The preset whose request is in flight (disables that slot + spins it).
    let presetBusy: Int?
    /// Step magnitude (1–8) for the directional pad; bound so the slider writes back.
    @Binding var speed: Double
    let enabled: Bool

    var body: some View {
        VStack(spacing: 16) {
            directionPad
            speedSlider
            presetRow
        }
    }

    // MARK: - Direction pad (3×3, diagonals included, empty centre)

    private var directionPad: some View {
        VStack(spacing: 8) {
            padRow([.upleft, .up, .upright])
            padRow([.left, nil, .right])
            padRow([.downleft, .down, .downright])
        }
        .frame(maxWidth: 260)
    }

    private func padRow(_ directions: [PTZDirection?]) -> some View {
        HStack(spacing: 8) {
            ForEach(Array(directions.enumerated()), id: \.offset) { _, direction in
                if let direction {
                    PTZPadButton(
                        direction: direction,
                        enabled: enabled,
                        onStep: { onStep(direction) }
                    )
                } else {
                    // Empty centre cell — keeps the arrows on a fixed grid.
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .fill(Color.clear)
                        .frame(width: 60, height: 60)
                        .overlay(
                            Image(systemName: "dot.scope")
                                .font(.system(size: 18, weight: .semibold))
                                .foregroundStyle(Theme.textSecondary.opacity(0.4))
                        )
                }
            }
        }
    }

    // MARK: - Speed slider (step magnitude, 1–8 — parity with web)

    private var speedSlider: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Label("Step speed", systemImage: "gauge.with.dots.needle.50percent")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(Theme.textSecondary)
                Spacer()
                Text("\(Int(speed))")
                    .font(.caption.weight(.semibold).monospacedDigit())
                    .foregroundStyle(Theme.accent)
            }
            Slider(value: $speed, in: 1 ... 8, step: 1)
                .tint(Theme.accent)
                .disabled(!enabled)
        }
        .frame(maxWidth: 260)
    }

    // MARK: - Presets (three slots)

    private var presetRow: some View {
        HStack(spacing: 10) {
            ForEach(1 ... 3, id: \.self) { index in
                PTZPresetButton(
                    index: index,
                    isSaved: savedPresets.contains(index),
                    isBusy: presetBusy == index,
                    enabled: enabled && presetBusy == nil,
                    onGoto: { onPresetGoto(index) },
                    onSet: { onPresetSet(index) },
                    onClear: { onPresetClear(index) }
                )
            }
        }
    }
}

// MARK: - Direction button (tap-to-step)

/// A single arrow on the pad. A plain tap fires `onStep` once — one small,
/// self-contained nudge in this direction. No hold, no release edge: this is
/// what replaced the runaway press-and-hold. A brief press-in scale gives tap
/// feedback.
private struct PTZPadButton: View {
    let direction: PTZDirection
    let enabled: Bool
    let onStep: () -> Void

    @State private var pressed = false

    var body: some View {
        Button(action: onStep) {
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Theme.accent.opacity(pressed ? 0.30 : 0.10))
                .frame(width: 60, height: 60)
                .overlay(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .stroke(Theme.accent.opacity(pressed ? 0.9 : 0.35),
                                lineWidth: pressed ? 2 : 1)
                )
                .overlay(
                    Image(systemName: symbol)
                        .font(.system(size: 22, weight: .bold))
                        .foregroundStyle(enabled ? Theme.accent : Theme.textSecondary)
                )
                .scaleEffect(pressed ? 0.92 : 1)
                .animation(.spring(duration: 0.2), value: pressed)
        }
        .buttonStyle(.plain)
        .contentShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        // Track the press purely for the tap-feedback highlight; the tap
        // action itself is delivered by the Button.
        .simultaneousGesture(
            DragGesture(minimumDistance: 0)
                .onChanged { _ in if enabled { pressed = true } }
                .onEnded { _ in pressed = false }
        )
        .disabled(!enabled)
        .accessibilityLabel(accessibilityLabel)
    }

    private var symbol: String {
        switch direction {
        case .up: return "chevron.up"
        case .down: return "chevron.down"
        case .left: return "chevron.left"
        case .right: return "chevron.right"
        case .upleft: return "arrow.up.left"
        case .upright: return "arrow.up.right"
        case .downleft: return "arrow.down.left"
        case .downright: return "arrow.down.right"
        }
    }

    private var accessibilityLabel: String {
        "Step " + direction.rawValue
    }
}

// MARK: - Preset button (tap = go, hold = save, ✕ = clear)

/// One preset slot. The fix for "clear worked but set didn't": the old code
/// hung `preset_set` off a `.simultaneousGesture(LongPressGesture)` on a plain
/// `Button`, where the button's own tap competed with the long-press and the
/// save rarely fired. Here a SINGLE `DragGesture(minimumDistance: 0)` drives
/// everything deterministically — press-down starts a hold timer that fires
/// `onSet` at 0.55s (and cancels the tap); a release before then recalls
/// (`onGoto`) when the slot is saved. No gesture composition to misfire. A
/// saved slot is filled + numbered and shows the ✕; an empty slot is dashed and
/// a quick tap does nothing (recall would only 502 an unset preset).
private struct PTZPresetButton: View {
    let index: Int
    let isSaved: Bool
    let isBusy: Bool
    let enabled: Bool
    let onGoto: () -> Void
    let onSet: () -> Void
    let onClear: () -> Void

    @State private var pressed = false
    @State private var isPressing = false
    @State private var didFireSet = false
    @State private var holdTask: Task<Void, Never>?

    private let holdThreshold: UInt64 = 550_000_000  // 0.55s

    var body: some View {
        ZStack(alignment: .topTrailing) {
            slotBody
                .frame(maxWidth: .infinity)
                .frame(height: 62)
                .background(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .fill(Theme.accent.opacity(isSaved ? 0.16 : 0.05))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .strokeBorder(
                            Theme.accent.opacity(isSaved ? 0.75 : 0.30),
                            style: StrokeStyle(lineWidth: isSaved ? 1.5 : 1,
                                               dash: isSaved ? [] : [4, 3])
                        )
                )
                .scaleEffect(pressed ? 0.95 : 1)
                .animation(.spring(duration: 0.2), value: pressed)
                .contentShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                .gesture(pressGesture)
                .disabled(!enabled)
                .accessibilityAddTraits(.isButton)
                .accessibilityLabel(
                    isSaved ? "Preset \(index), saved. Tap to recall, hold to overwrite."
                            : "Preset \(index), empty. Hold to save the current position."
                )
                .accessibilityAction(named: "Save current position") {
                    if enabled { onSet() }
                }
                .accessibilityAction(named: isSaved ? "Recall" : "Recall (empty)") {
                    if enabled && isSaved { onGoto() }
                }

            if isSaved && !isBusy {
                Button(action: { if enabled { onClear() } }) {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundStyle(Theme.textSecondary)
                        .padding(4)
                }
                .buttonStyle(.plain)
                .disabled(!enabled)
                .accessibilityLabel("Clear preset \(index)")
            }
        }
        .accessibilityElement(children: .contain)
    }

    /// One gesture: track the touch, arm a hold timer on press-down, decide
    /// tap-vs-hold on release.
    private var pressGesture: some Gesture {
        DragGesture(minimumDistance: 0)
            .onChanged { _ in
                guard enabled, !isPressing else { return }
                isPressing = true
                didFireSet = false
                pressed = true
                holdTask = Task { @MainActor in
                    try? await Task.sleep(nanoseconds: holdThreshold)
                    guard !Task.isCancelled, isPressing else { return }
                    didFireSet = true
                    pressed = false
                    onSet()
                }
            }
            .onEnded { _ in
                holdTask?.cancel()
                holdTask = nil
                let wasHeld = didFireSet
                pressed = false
                isPressing = false
                didFireSet = false
                // Quick release before the hold fired = a tap → recall a saved
                // slot (an empty slot does nothing).
                if enabled, !wasHeld, isSaved {
                    onGoto()
                }
            }
    }

    @ViewBuilder
    private var slotBody: some View {
        VStack(spacing: 3) {
            if isBusy {
                ProgressView().tint(Theme.accent)
            } else {
                Image(systemName: isSaved ? "\(index).circle.fill" : "\(index).circle")
                    .font(.system(size: 22, weight: .semibold))
                    .foregroundStyle(isSaved ? Theme.accent : Theme.textSecondary)
            }
            Text(isSaved ? "Preset" : "Empty")
                .font(.caption2.weight(.semibold))
                .foregroundStyle(isSaved ? Theme.accent : Theme.textSecondary)
            Text(isSaved ? "tap · hold" : "hold to save")
                .font(.system(size: 9, weight: .medium))
                .foregroundStyle(Theme.textSecondary.opacity(0.8))
        }
    }
}
