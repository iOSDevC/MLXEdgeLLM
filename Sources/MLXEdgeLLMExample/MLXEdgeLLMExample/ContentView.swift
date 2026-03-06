import SwiftUI
import PhotosUI
import MLXEdgeLLM

// MARK: - View
struct ContentView: View {
    @StateObject private var vm = DemoViewModel()
    @State private var pickerItem: PhotosPickerItem?
    @State private var selectedVisionModel: VisionModel = .qwen35_0_8b
    @State private var visionRunMode: VisionRunMode = .standard
    @State private var selectedSpecializedModel: SpecializedVisionModel = .fastVLM_0_5b_fp16
    
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
                                vm.selectedImage = PlatformImage(data: data)
                            }
                        }
                    }
                    
                    if let img = vm.selectedImage {
                        // Standard VLMs
                        GroupBox("TextModel VLMs") {
                            HStack {
                                VStack(spacing: 16) {
                                    Picker("Model", selection: $selectedVisionModel) {
                                        ForEach(VisionModel.allCases, id: \.self) { model in
                                            Text(model.displayName).tag(model)
                                        }
                                    }
                                    
                                    Picker("Mode", selection: $visionRunMode) {
                                        ForEach(VisionRunMode.allCases) { mode in
                                            Text(mode.rawValue).tag(mode)
                                        }
                                    }
                                    .pickerStyle(.segmented)
                                }
                                Spacer()
                                btn("Run", .blue) {
                                    switch visionRunMode {
                                        case .standard:
                                            await vm.runVLM(model: selectedVisionModel, image: img)
                                        case .stream:
                                            await vm.runStreamVLM(model: selectedVisionModel, image: img)
                                    }
                                }
                                .frame(maxWidth: 120)
                            }
                            .frame(maxWidth: .infinity)
                        }
                        
                        // Specialized OCR
                        GroupBox("Specialized OCR") {
                            HStack {
                                Picker("Model", selection: $selectedSpecializedModel) {
                                    ForEach(SpecializedVisionModel.allCases, id: \.self) { model in
                                        Text(specializedModelLabel(model)).tag(model)
                                    }
                                }
                                
                                Spacer()
                                
                                btn("Run OCR", .orange) {
                                    await vm.runSpecialized(model: selectedSpecializedModel, image: img)
                                }
                                .frame(maxWidth: 120)
                            }
                            .frame(maxWidth: .infinity)
                            
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
                    btn("💬 Text Chat(Qwen3 1.7B)", .green) { await vm.runTextChat() }
                    
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
                        .frame(maxHeight: .infinity)
                        .background(Color.outputBackground)
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
            Image(platformImage: img)
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
    
    private func specializedModelLabel(_ model: SpecializedVisionModel) -> String {
        switch model {
            case .fastVLM_0_5b_fp16:
                return "⚡ FastVLM 0.5B"
            case .graniteDocling_258m:
                return "📄 Granite 258M"
            default:
                return String(describing: model)
        }
    }
    
    private enum VisionRunMode: String, CaseIterable, Identifiable {
        case standard = "Standard"
        case stream = "Stream"
        
        var id: String { rawValue }
    }
}

#Preview { ContentView() }
