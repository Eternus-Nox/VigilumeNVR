import SwiftUI

/// Draw the region of a camera where faces (or plates) are worth recognizing,
/// over a heatmap of where they have actually been legible.
///
/// WHY THE HEATMAP IS THE POINT
/// ============================
/// Asked to draw a face zone on a still frame, almost everyone draws the region
/// where people WALK. That is the obvious guess and frequently the wrong one:
/// whether a face is legible depends on range, lens, mounting height and where
/// the light is, and none of those are visible in a still. The region that
/// matters is where something READABLE has actually come from, and that is
/// knowable only from history.
///
/// So the server accumulates two numbers per grid cell — how often a face was
/// centred there, and the mean legibility of those sightings — and this screen
/// renders them on different visual channels:
///
///     DENSITY  -> opacity   (how much evidence there is)
///     QUALITY  -> hue       (whether anything readable came from there)
///
/// That separation is the whole design. A busy, unreadable strip of far
/// pavement shows as strong-but-red; a quiet, sharp doorstep shows as
/// faint-but-green. Both are invisible on the frame itself, and the second one
/// is where the zone belongs.
///
/// Colour is never the only channel: every cell also carries its density as
/// opacity, the legend states the mapping in words, and "Use suggestion" gives
/// a colour-free path to a sensible region.
struct RecognitionZoneEditor: View {
    let camera: Camera
    /// "face" | "plate".
    let kind: String
    var onSaved: () async -> Void

    @EnvironmentObject private var session: SessionModel
    @Environment(\.dismiss) private var dismiss

    @State private var points: [CGPoint] = []
    @State private var heatmap: RecognitionHeatmap?
    @State private var loading = true
    @State private var saving = false
    @State private var showHeatmap = true
    @State private var alert: EditorAlert?

    private enum EditorAlert: Identifiable {
        case error(String)
        case confirmClearMap

        var id: String {
            switch self {
            case .error(let m): return "error:\(m)"
            case .confirmClearMap: return "clear"
            }
        }
    }

    private var isFace: Bool { kind == "face" }
    private var title: String { isFace ? "Face Zone" : "Plate Zone" }

    private var existing: [IncludeZone] {
        (isFace ? camera.faceZones : camera.plateZones) ?? []
    }

    var body: some View {
        VStack(spacing: 0) {
            canvas
            controls
        }
        .background(Theme.bg)
        .navigationTitle(title)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button("Save") { Task { await save() } }
                    .fontWeight(.semibold)
                    .disabled(saving || (points.count > 0 && points.count < 3))
            }
        }
        .alert(
            alertTitle,
            isPresented: Binding(
                get: { alert != nil },
                set: { if !$0 { alert = nil } }
            ),
            presenting: alert
        ) { which in
            switch which {
            case .error:
                Button("OK", role: .cancel) {}
            case .confirmClearMap:
                Button("Clear", role: .destructive) { Task { await clearMap() } }
                Button("Cancel", role: .cancel) {}
            }
        } message: { which in
            switch which {
            case .error(let message):
                Text(message)
            case .confirmClearMap:
                Text("Forget where \(isFace ? "faces" : "plates") have been seen on this camera. Do this after moving or re-aiming it, when the old view's history no longer describes what it sees.")
            }
        }
        .task { await load() }
    }

    private var alertTitle: String {
        guard let alert else { return "" }
        switch alert {
        case .error: return "Something went wrong"
        case .confirmClearMap: return "Clear the heatmap?"
        }
    }

    // MARK: Canvas

    private var canvas: some View {
        GeometryReader { geo in
            let size = geo.size
            ZStack {
                // The camera's own view, so the zone is drawn against what it
                // actually sees rather than an abstract rectangle.
                AsyncImage(url: session.api?.cameraSnapshotURL(camera.name)) { phase in
                    if case .success(let image) = phase {
                        image.resizable().scaledToFit()
                    } else {
                        Rectangle().fill(Theme.bgDeep)
                    }
                }

                if showHeatmap, let heatmap, !heatmap.isEmpty {
                    HeatmapOverlay(heatmap: heatmap)
                        .allowsHitTesting(false)
                }

                ZonePolygon(points: points, size: size)
                    .allowsHitTesting(false)

                ForEach(Array(points.enumerated()), id: \.offset) { index, p in
                    Circle()
                        .fill(Theme.accent)
                        .frame(width: 14, height: 14)
                        .overlay(Circle().stroke(.white, lineWidth: 2))
                        .position(x: p.x * size.width, y: p.y * size.height)
                        .gesture(
                            DragGesture()
                                .onChanged { value in
                                    points[index] = CGPoint(
                                        x: min(max(value.location.x / size.width, 0), 1),
                                        y: min(max(value.location.y / size.height, 0), 1)
                                    )
                                }
                        )
                        .accessibilityLabel("Zone corner \(index + 1)")
                }
            }
            .contentShape(Rectangle())
            .onTapGesture { location in
                points.append(CGPoint(
                    x: min(max(location.x / size.width, 0), 1),
                    y: min(max(location.y / size.height, 0), 1)
                ))
            }
            .overlay {
                if loading { ProgressView().tint(Theme.accent) }
            }
        }
        .aspectRatio(4.0 / 3.0, contentMode: .fit)
        .clipped()
    }

    // MARK: Controls

    private var controls: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                if let heatmap, !heatmap.isEmpty {
                    HeatmapLegend(samples: heatmap.samples)
                } else if !loading {
                    Label {
                        Text("No sightings recorded yet. Leave recognition running for a while and a map of where \(isFace ? "faces" : "plates") are actually readable will build up here.")
                            .font(.caption)
                            .foregroundStyle(Theme.textSecondary)
                    } icon: {
                        Image(systemName: "clock.arrow.circlepath")
                            .foregroundStyle(Theme.textSecondary)
                    }
                }

                HStack(spacing: 10) {
                    Toggle("Heatmap", isOn: $showHeatmap)
                        .toggleStyle(.button)
                        .tint(Theme.accent)
                        .disabled(heatmap?.isEmpty ?? true)

                    Button("Undo point") { if !points.isEmpty { points.removeLast() } }
                        .disabled(points.isEmpty)

                    Button("Clear") { points = [] }
                        .disabled(points.isEmpty)
                }
                .font(.footnote)
                .buttonStyle(.bordered)
                .tint(Theme.accent)

                if let suggestion = heatmap?.suggestedZone, suggestion.count >= 3 {
                    Button {
                        points = suggestion.map { CGPoint(x: $0[0], y: $0[1]) }
                    } label: {
                        Label("Use suggested region", systemImage: "wand.and.stars")
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(Theme.accent)
                    .font(.footnote)
                }

                Text(instructions)
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)

                Button(role: .destructive) {
                    alert = .confirmClearMap
                } label: {
                    Label("Clear heatmap history", systemImage: "trash")
                }
                .font(.caption)
                .disabled(heatmap?.isEmpty ?? true)
            }
            .padding(16)
        }
    }

    private var instructions: String {
        if points.isEmpty {
            return "Tap to place corners around the area where \(isFace ? "a face" : "a plate") is readable. Leave it empty to search the whole frame — correct, just slower. Drag a corner to adjust it."
        }
        if points.count < 3 {
            return "A zone needs at least three corners."
        }
        return "Drag any corner to adjust. Saving replaces this camera's \(isFace ? "face" : "plate") zone."
    }

    // MARK: Actions

    private func load() async {
        guard let api = session.api else { return }
        loading = true
        defer { loading = false }
        points = existing.first?.points.map { CGPoint(x: $0[0], y: $0[1]) } ?? []
        do {
            heatmap = try await api.recognitionHeatmap(camera: camera.name, kind: kind)
        } catch {
            // A missing heatmap is not an error worth interrupting for — the
            // editor still works, it just cannot advise.
            heatmap = nil
        }
    }

    private func save() async {
        guard let api = session.api else { return }
        guard points.isEmpty || points.count >= 3 else { return }
        saving = true
        defer { saving = false }
        let zone = points.isEmpty
            ? []
            : [IncludeZone(name: isFace ? "face" : "plate",
                           points: points.map { [Double($0.x), Double($0.y)] })]
        do {
            _ = try await api.updateCameraZones(
                camera: camera,
                faceZones: isFace ? zone : nil,
                plateZones: isFace ? nil : zone
            )
            await onSaved()
            dismiss()
        } catch {
            // A cancelled request is not a failure — SwiftUI cancels the
            // .task when the view refreshes. Alerting on it turns pull-to-
            // refresh into a scary "is the NVR reachable?".
            if (error as? ApiError)?.isCancelled == true { return }
            alert = .error((error as? ApiError)?.message ?? error.localizedDescription)
        }
    }

    private func clearMap() async {
        guard let api = session.api else { return }
        do {
            try await api.clearRecognitionHeatmap(camera: camera.name, kind: kind)
            heatmap = try? await api.recognitionHeatmap(camera: camera.name, kind: kind)
        } catch {
            // A cancelled request is not a failure — SwiftUI cancels the
            // .task when the view refreshes. Alerting on it turns pull-to-
            // refresh into a scary "is the NVR reachable?".
            if (error as? ApiError)?.isCancelled == true { return }
            alert = .error((error as? ApiError)?.message ?? error.localizedDescription)
        }
    }
}

// MARK: - Overlay

/// The legibility map, drawn as one Canvas rather than hundreds of views.
///
/// 768 cells as SwiftUI shapes would be 768 view identities to diff on every
/// layout pass; a single Canvas draws the same thing in one pass and scrolls
/// smoothly on a phone.
struct HeatmapOverlay: View {
    let heatmap: RecognitionHeatmap

    var body: some View {
        Canvas { context, size in
            let cw = size.width / CGFloat(heatmap.cols)
            let ch = size.height / CGFloat(heatmap.rows)
            for row in 0 ..< heatmap.rows {
                for col in 0 ..< heatmap.cols {
                    let density = heatmap.count(col: col, row: row)
                    guard density > 0.02 else { continue }
                    let quality = heatmap.meanQuality(col: col, row: row)
                    let rect = CGRect(
                        x: CGFloat(col) * cw, y: CGFloat(row) * ch,
                        width: cw + 0.5, height: ch + 0.5
                    )
                    context.fill(
                        Path(rect),
                        with: .color(Self.tint(quality: quality).opacity(0.18 + 0.55 * density))
                    )
                }
            }
        }
    }

    /// Hue carries LEGIBILITY, not traffic. Red does not mean "busy" here — it
    /// means "nothing readable has ever come from this cell", which is the
    /// opposite of what a traffic heatmap's red would say, and is why the
    /// legend spells it out rather than relying on convention.
    static func tint(quality: Double) -> Color {
        if quality >= 0.65 { return Theme.success }
        if quality >= 0.45 { return Theme.warning }
        return Theme.danger
    }
}

/// States the colour mapping in words, so the overlay does not depend on
/// colour perception or on guessing which way round the scale runs.
struct HeatmapLegend: View {
    let samples: Int

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("\(samples) sighting\(samples == 1 ? "" : "s") recorded")
                .font(.caption.weight(.semibold))
                .foregroundStyle(Theme.textPrimary)
            HStack(spacing: 12) {
                swatch(Theme.success, "Readable")
                swatch(Theme.warning, "Marginal")
                swatch(Theme.danger, "Not readable")
            }
            Text("Colour shows how legible sightings were; stronger shading means more of them. Put the zone where sightings are readable — not simply where they are frequent.")
                .font(.caption2)
                .foregroundStyle(Theme.textSecondary)
        }
    }

    private func swatch(_ color: Color, _ label: String) -> some View {
        HStack(spacing: 4) {
            RoundedRectangle(cornerRadius: 3).fill(color).frame(width: 12, height: 12)
            Text(label).font(.caption2).foregroundStyle(Theme.textSecondary)
        }
    }
}

/// The zone being drawn. Closed once it has three corners; an open path before
/// that, so a half-finished zone looks unfinished rather than wrong.
struct ZonePolygon: View {
    let points: [CGPoint]
    let size: CGSize

    var body: some View {
        Path { path in
            guard let first = points.first else { return }
            path.move(to: CGPoint(x: first.x * size.width, y: first.y * size.height))
            for p in points.dropFirst() {
                path.addLine(to: CGPoint(x: p.x * size.width, y: p.y * size.height))
            }
            if points.count >= 3 { path.closeSubpath() }
        }
        .fill(Theme.accent.opacity(points.count >= 3 ? 0.20 : 0))
        .overlay {
            Path { path in
                guard let first = points.first else { return }
                path.move(to: CGPoint(x: first.x * size.width, y: first.y * size.height))
                for p in points.dropFirst() {
                    path.addLine(to: CGPoint(x: p.x * size.width, y: p.y * size.height))
                }
                if points.count >= 3 { path.closeSubpath() }
            }
            .stroke(Theme.accent, lineWidth: 2)
        }
    }
}
