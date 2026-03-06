import Foundation
import MLX
import MLXVLM
import MLXLMCommon

#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

// MARK: - SpecializedVisionEngine

/// Internal @MainActor class for FastVLM 0.5B and Granite Docling 258M.
///
/// Both models ship MLX checkpoints that use architectures compatible with
/// VLMModelFactory (LLaVA-family base + fp16 weights). We load them the same
/// way as standard VLMs — via VLMModelFactory — which handles architecture
/// dispatch automatically from the model's config.json.
@MainActor
final class SpecializedVisionEngine {

    // MARK: - State

    private var modelContainer: ModelContainer?
    private let model: SpecializedVisionModel

    /// Low temperature for deterministic OCR output
    private let generateParameters: GenerateParameters

    // MARK: - Init

    init(model: SpecializedVisionModel) {
        self.model = model
        // Temperature 0.0 for deterministic document/receipt extraction
        self.generateParameters = GenerateParameters(temperature: 0.0)
    }

    // MARK: - Load

    func load(onProgress: @escaping (String) -> Void) async throws {
        guard modelContainer == nil else { return }

        // Smaller cache since models are 270–420 MB
        MLX.Memory.cacheLimit = 48 * 1024 * 1024

        let config = ModelConfiguration(id: model.rawValue)
        modelContainer = try await VLMModelFactory.shared.loadContainer(
            configuration: config
        ) { progress in
            let pct = Int(progress.fractionCompleted * 100)
            Task { @MainActor in
                onProgress("Downloading \(self.model.displayName): \(pct)%")
            }
        }
        onProgress("\(model.displayName) ready ✓")
    }

    // MARK: - Generate

    func generate(
        prompt: String,
        image: PlatformImage?,
        maxTokens: Int = 512,
        onToken: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        guard let container = modelContainer else {
            throw MLXEdgeLLMError.modelNotLoaded
        }

        // Build UserInput — VLMs accept .url for images
        let userInput: UserInput
        if let img = image, let url = saveImageToTemp(img) {
            userInput = UserInput(prompt: prompt, images: [.url(url)])
        } else {
            userInput = UserInput(prompt: prompt)
        }

        return try await container.perform { context in
            let input = try await context.processor.prepare(input: userInput)
            let result = try MLXLMCommon.generate(
                input: input,
                parameters: self.generateParameters,
                context: context
            ) { tokens in
                let partial = context.tokenizer.decode(tokens: tokens)
                Task { @MainActor in onToken(partial) }
                return tokens.count >= maxTokens ? .stop : .more
            }
            return context.tokenizer.decode(tokens: result.tokens)
        }
    }

    // MARK: - Helpers

    private func saveImageToTemp(_ image: PlatformImage) -> URL? {
        let url = FileManager.default.temporaryDirectory
            .appending(path: "mlxedge_spec_\(UUID().uuidString).jpg")
        #if canImport(UIKit)
        guard let data = image.jpegData(compressionQuality: 0.95) else { return nil }
        #else
        guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil),
              let data = NSBitmapImageRep(cgImage: cgImage)
                  .representation(using: .jpeg, properties: [:]) else { return nil }
        #endif
        try? data.write(to: url)
        return url
    }
}
