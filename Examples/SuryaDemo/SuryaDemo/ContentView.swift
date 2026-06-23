import MLXSurya
import SwiftUI

struct ContentView: View {
    @State private var model: SuryaParserViewModel

    @MainActor init() { _model = State(initialValue: SuryaParserViewModel()) }
    @MainActor init(model: SuryaParserViewModel) { _model = State(initialValue: model) }

    var body: some View {
        HSplitView {
            controls
                .frame(minWidth: 280, maxWidth: 360, maxHeight: .infinity)
            preview
                .frame(minWidth: 320, maxHeight: .infinity)
            ScrollView {
                Text((model.current?.text).flatMap { $0.isEmpty ? nil : $0 } ?? "Results appear here.")
                    .font(.system(.body, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
            }
            .frame(minWidth: 280, maxHeight: .infinity)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var controls: some View {
        @Bindable var model = model
        return Form {
            Section("Model (surya-ocr-2 VLM)") {
                Button(action: model.downloadModel) {
                    Label(
                        model.isModelCached
                            ? "surya-ocr-2 cached" : "Download surya-ocr-2 (~1.4 GB)",
                        systemImage: model.isModelCached ? "internaldrive" : "arrow.down.circle")
                }
                .disabled(model.isDownloading || model.isRunning || model.isModelCached)
                if model.isDownloading {
                    ProgressView(value: model.downloadProgress)
                    Text("\(Int(model.downloadProgress * 100))%")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            Section("Input") {
                VStack(alignment: .leading, spacing: 4) {
                    Button("PDF / image", action: model.pickInput)
                    Text(model.inputPath.isEmpty ? "—" : model.inputPath)
                        .font(.caption).foregroundStyle(.secondary)
                        .lineLimit(1).truncationMode(.middle)
                }
            }
            Section("Mode") {
                Picker("Stage", selection: $model.mode) {
                    ForEach(DemoMode.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.radioGroup)
            }
            Section("VLM precision") {
                Picker("Precision", selection: $model.precision) {
                    Text("bf16 (full)").tag(SuryaPrecision.bf16)
                    Text("int8 (less mem)").tag(SuryaPrecision.int8)
                }
                .pickerStyle(.segmented)
            }
            Section("Image detail (OCR/Layout/Table speed)") {
                Picker("Detail", selection: $model.imageDetail) {
                    ForEach(ImageDetail.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                Text("Lower detail = fewer image tokens = faster, less accurate on small text.")
                    .font(.caption2).foregroundStyle(.tertiary)
            }
            Section {
                HStack {
                    Button(action: model.run) { Label("Run", systemImage: "play.fill") }
                        .disabled(!model.canRun)
                    if model.isRunning {
                        Button("Cancel", role: .cancel, action: model.cancel)
                        ProgressView().controlSize(.small)
                    }
                }
                Text(model.status).font(.callout).foregroundStyle(.secondary)
                Text("First run downloads model weights. Debug builds are slow — build Release.")
                    .font(.caption2).foregroundStyle(.tertiary)
            }
        }
        .formStyle(.grouped)
    }

    @ViewBuilder private var preview: some View {
        VStack(spacing: 0) {
            if let page = model.current {
                pager
                BoxOverlay(image: page.image, imageSize: page.imageSize, boxes: page.boxes)
                    .padding()
            } else {
                ContentUnavailableView("No page", systemImage: "doc.text.image")
            }
        }
    }

    /// Page flipper: ◀ / "Page X of N" / ▶ over the completed pages.
    private var pager: some View {
        HStack {
            Button(action: model.goPrev) { Image(systemName: "chevron.left") }
                .disabled(!model.canPrev)
            Spacer()
            Text(pagerLabel).font(.callout).monospacedDigit()
            Spacer()
            Button(action: model.goNext) { Image(systemName: "chevron.right") }
                .disabled(!model.canNext)
        }
        .padding(.horizontal).padding(.vertical, 6)
        .background(.bar)
    }

    private var pagerLabel: String {
        let total = max(model.totalPages, model.results.count)
        let done = model.results.count
        let suffix = (model.isRunning && done < total) ? " · \(done) ready" : ""
        return "Page \(model.selectedPage + 1) of \(total)\(suffix)"
    }
}

/// Draws the page image with detected/layout boxes overlaid, scaled from source-image pixels to
/// the aspect-fit display rect.
struct BoxOverlay: View {
    let image: CGImage
    let imageSize: CGSize
    let boxes: [DemoBox]

    var body: some View {
        GeometryReader { geo in
            let scale = min(geo.size.width / imageSize.width, geo.size.height / imageSize.height)
            let drawn = CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
            let ox = (geo.size.width - drawn.width) / 2
            let oy = (geo.size.height - drawn.height) / 2
            Image(decorative: image, scale: 1)
                .resizable()
                .frame(width: drawn.width, height: drawn.height)
                .position(x: geo.size.width / 2, y: geo.size.height / 2)
            Canvas { ctx, _ in
                func disp(_ p: CGPoint) -> CGPoint {
                    CGPoint(x: ox + p.x * scale, y: oy + p.y * scale)
                }
                // Boxes.
                for box in boxes {
                    guard let first = box.points.first else { continue }
                    var path = Path()
                    path.move(to: disp(first))
                    for p in box.points.dropFirst() { path.addLine(to: disp(p)) }
                    path.closeSubpath()
                    ctx.stroke(path, with: .color(.red.opacity(0.85)), lineWidth: 1.5)
                }
                // Reading-order flow line + numbered markers (Layout / OCR modes).
                let ordered = boxes.filter { $0.order != nil }
                    .sorted { ($0.order ?? 0) < ($1.order ?? 0) }
                if ordered.count > 1 {
                    var flow = Path()
                    flow.move(to: disp(ordered[0].center))
                    for b in ordered.dropFirst() { flow.addLine(to: disp(b.center)) }
                    ctx.stroke(
                        flow, with: .color(.blue.opacity(0.5)),
                        style: StrokeStyle(lineWidth: 1.5, dash: [4, 3]))
                }
                for b in ordered {
                    guard let o = b.order else { continue }
                    let c = disp(b.center)
                    let r: CGFloat = 9
                    ctx.fill(
                        Path(ellipseIn: CGRect(x: c.x - r, y: c.y - r, width: 2 * r, height: 2 * r)),
                        with: .color(.blue.opacity(0.85)))
                    ctx.draw(
                        Text("\(o)").font(.system(size: 10, weight: .bold)).foregroundStyle(.white),
                        at: c)
                }
            }
        }
    }
}

#Preview { ContentView().frame(width: 900, height: 600) }
