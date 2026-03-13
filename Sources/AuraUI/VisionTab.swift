import SwiftUI
import AuraCore
import PhotosUI
// MARK: - Vision Tab

public struct VisionTab: View {
    @StateObject private var vm = VisionViewModel()
    @State private var pickerItem: PhotosPickerItem?
    @State private var selectedModel: Model = .qwen35_0_8b
    @State private var runMode: AuraLocal.VisionRunMode = .standard
    @State private var customPrompt: String = "Describe this image in detail."
    
    public init() {}
    
    public var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    
                    // Image picker
                    PhotosPicker(selection: $pickerItem, matching: .images) {
                        ImagePickerPreview(image: vm.selectedImage)
                    }
                    .onChange(of: pickerItem) { _, item in
                        Task {
                            if let data = try? await item?.loadTransferable(type: Data.self) {
                                vm.selectedImage = PlatformImage(data: data)
                                vm.output = ""
                            }
                        }
                    }
                    
                    // Controls
                    GroupBox {
                        VStack(spacing: 12) {
                            LabeledPicker("Model", selection: $selectedModel, items: Model.visionModels)
                            
                            Picker("Mode", selection: $runMode) {
                                ForEach(AuraLocal.VisionRunMode.allCases) { mode in
                                    Text(mode.rawValue).tag(mode)
                                }
                            }
                            .pickerStyle(.segmented)
                            
                            TextField("Custom prompt...", text: $customPrompt, axis: .vertical)
                                .lineLimit(2...4)
                                .font(.subheadline)
                                .padding(8)
                                .background(Color.tertiaryGroupedBackground, in: RoundedRectangle(cornerRadius: 8))
                        }
                    } label: {
                        Label("Settings", systemImage: "slider.horizontal.3")
                            .font(.subheadline.weight(.semibold))
                    }
                    
                    // Run button
                    RunButton(
                        title: "Analyze Image",
                        subtitle: selectedModel.displayName,
                        isDownloaded: selectedModel.isDownloaded,
                        isLoading: vm.isLoading,
                        color: .blue
                    ) {
                        guard let img = vm.selectedImage else { return }
                        await vm.run(
                            model: selectedModel,
                            image: img,
                            prompt: customPrompt,
                            mode: runMode
                        )
                    }
                    .disabled(vm.selectedImage == nil)
                    
                    // Progress + Output
                    StatusSection(progress: vm.progress, output: vm.output)
                }
                .padding()
            }
            .navigationTitle("Vision")
            .background(Color.groupedBackground)
        }
    }
}
#Preview {
    VisionTab()
}
