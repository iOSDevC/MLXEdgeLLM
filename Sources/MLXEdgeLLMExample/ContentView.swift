import SwiftUI
import PhotosUI
import MLXEdgeLLM

@MainActor
final class DemoViewModel: ObservableObject {
    @Published var output: String = ""
    @Published var progress: String = ""
    @Published var isLoading = false
    @Published var selectedImage: UIImage?

    func runVLM(model: VisionModel, image: UIImage) async {
        startLoading()
        do {
            let vlm = try await MLXEdgeLLMVision(model: model) { p in
                Task { @MainActor in self.progress = p }
            }
            let result = try await vlm.extractReceipt(image)
            output = "📋 \(model.displayName):\n\n\(result)"
        } catch { output = "❌ \(error.localizedDescription)" }
        stopLoading()
    }

    func runSpecialized(model: SpecializedVisionModel, image: UIImage) async {
        startLoading()
        do {
            let e = try await MLXEdgeLLMSpecialized(model: model) { p in
                Task { @MainActor in self.progress = p }
            }
            var result = try await e.extractDocument(image)
            if model.outputsDocTags {
                result = MLXEdgeLLMSpecialized.parseDocTags(result)
                output = "📝 Granite → Markdown:\n\n\(result)"
            } else {
                output = "⚡ FastVLM JSON:\n\n\(result)"
            }
        } catch { output = "❌ \(error.localizedDescription)" }
        stopLoading()
    }

    func runStreamVLM(model: VisionModel, image: UIImage) async {
        startLoading()
        output = ""
        do {
            let vlm = try await MLXEdgeLLMVision(model: model) { p in
                Task { @MainActor in self.progress = p }
            }
            for try await token in vlm.stream("Describe this receipt in detail.", image: image) {
                output += token
            }
        } catch { output = "❌ \(error.localizedDescription)" }
        stopLoading()
    }

    func runTextChat() async {
        startLoading()
        do {
            let llm = try await MLXEdgeLLM(model: .qwen3_1_7b) { p in
                Task { @MainActor in self.progress = p }
            }
            output = ""
            for try await token in llm.stream("¿Cuánto es el IVA en México y cómo aparece en tickets?") {
                output += token
            }
        } catch { output = "❌ \(error.localizedDescription)" }
        stopLoading()
    }

    private func startLoading() { isLoading = true; progress = "" }
    private func stopLoading() { isLoading = false; progress = "" }
}

// MARK: - View

struct ContentView: View {
    @StateObject private var vm = DemoViewModel()
    @State private var pickerItem: PhotosPickerItem?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 14) {

                    // Image picker
                    PhotosPicker(selection: $pickerItem, matching: .images) {
                        imagePreview
                    }
                    .onChange(of: pickerItem) { _, item in
                        Task {
                            if let data = try? await item?.loadTransferable(type: Data.self) {
                                vm.selectedImage = UIImage(data: data)
                            }
                        }
                    }

                    if let img = vm.selectedImage {
                        // Standard VLMs
                        GroupBox("Standard VLMs") {
                            HStack {
                                btn("Qwen3.5 0.8B\nOCR", .blue) { await vm.runVLM(model: .qwen35_0_8b, image: img) }
                                btn("SmolVLM\n500M", .blue) { await vm.runVLM(model: .smolvlm_500m, image: img) }
                                btn("Stream\nQwen3.5", .cyan) { await vm.runStreamVLM(model: .qwen35_0_8b, image: img) }
                            }
                        }

                        // Specialized OCR
                        GroupBox("Specialized OCR") {
                            HStack {
                                btn("⚡ FastVLM\n0.5B", .orange) { await vm.runSpecialized(model: .fastVLM_0_5b, image: img) }
                                btn("📄 Granite\n258M", .purple) { await vm.runSpecialized(model: .graniteDocling_258m, image: img) }
                            }
                            Text("FastVLM: JSON · Granite: DocTags→Markdown")
                                .font(.caption2).foregroundStyle(.secondary)
                        }
                    } else {
                        Text("Select a receipt or document image above")
                            .foregroundStyle(.secondary)
                            .font(.subheadline)
                            .padding()
                    }

                    // Text chat (no image needed)
                    btn("💬 Text Chat (Qwen3 1.7B)", .green) { await vm.runTextChat() }

                    // Progress
                    if !vm.progress.isEmpty {
                        Text(vm.progress).font(.caption).foregroundStyle(.secondary)
                    }

                    // Output
                    if !vm.output.isEmpty {
                        ScrollView {
                            Text(vm.output)
                                .font(.system(.caption, design: .monospaced))
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(8)
                        }
                        .frame(maxHeight: 240)
                        .background(Color(.systemGroupedBackground))
                        .cornerRadius(8)
                    }
                }
                .padding()
            }
            .navigationTitle("MLXEdgeLLM")
            .overlay(loadingOverlay)
        }
    }

    @ViewBuilder
    private var imagePreview: some View {
        if let img = vm.selectedImage {
            Image(uiImage: img)
                .resizable().scaledToFit()
                .frame(maxHeight: 200)
                .cornerRadius(10)
        } else {
            RoundedRectangle(cornerRadius: 10)
                .fill(Color.secondary.opacity(0.1))
                .frame(height: 120)
                .overlay {
                    Label("Tap to select image", systemImage: "photo.badge.plus")
                        .foregroundStyle(.secondary)
                }
        }
    }

    @ViewBuilder
    private var loadingOverlay: some View {
        if vm.isLoading {
            ProgressView()
                .padding(20)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
        }
    }

    private func btn(_ title: String, _ color: Color, action: @escaping () async -> Void) -> some View {
        Button { Task { await action() } } label: {
            Text(title)
                .font(.subheadline.weight(.medium))
                .multilineTextAlignment(.center)
                .frame(maxWidth: .infinity, minHeight: 44)
                .padding(.vertical, 6)
                .background(color.opacity(0.12))
                .foregroundStyle(color)
                .cornerRadius(8)
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(color.opacity(0.3)))
        }
    }
}

#Preview { ContentView() }
