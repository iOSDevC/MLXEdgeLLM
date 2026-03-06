import Foundation
import MLX
import MLXVLM
import MLXLMCommon
import SwiftUI
#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

// MARK: - VisionEngine

/// Internal @MainActor class for VLM inference via mlx-swift-lm's MLXVLM.
/// Uses VLMModelFactory.shared.loadContainer + ModelContainer.perform.
@MainActor
final class VisionEngine {

    // MARK: - State

    private var modelContainer: ModelContainer?
    private let model: VisionModel
    private let generateParameters: GenerateParameters

    // MARK: - Init

    init(model: VisionModel, temperature: Float = 0.1) {
        self.model = model
        self.generateParameters = GenerateParameters(temperature: temperature)
    }

    // MARK: - Load

    func load(onProgress: @escaping (String) -> Void) async throws {
        guard modelContainer == nil else { return }

        MLX.Memory.cacheLimit = 64 * 1024 * 1024

        let config = ModelConfiguration(id: model.rawValue)
        modelContainer = try await VLMModelFactory.shared.loadContainer(
            configuration: config
        ) { progress in
            let pct = Int(progress.fractionCompleted * 100)
            Task { @MainActor in
                onProgress("Downloading \(progress.fileCompletedCount)/\(progress.fileTotalCount) — \(pct)%")
            }
        }
        onProgress("Model ready.")
    }

    // MARK: - Generate

    /// Run vision + text through the model. `image` is optional for text-only queries.
    /// Calls `onToken` with each partial text as tokens arrive.
    func generate(
        prompt: String,
        image: PlatformImage?,
        maxTokens: Int = 800,
        onToken: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        guard let container = modelContainer else {
            throw MLXEdgeLLMError.modelNotLoaded
        }

        // Build UserInput
        let userInput: UserInput
        if let img = image, let url = saveImageToTemp(img) {
            userInput = UserInput(
                prompt: prompt,
                images: [.url(url)]
            )
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
            // decode final dentro del closure donde `context` sí existe
            return context.tokenizer.decode(tokens: result.tokens)
        }
    }

    // MARK: - Helpers

    /// Save UIImage/NSImage to a temp file and return its URL for UserInput.
    private func saveImageToTemp(_ image: PlatformImage) -> URL? {
        let url = FileManager.default.temporaryDirectory
            .appending(path: "mlxedge_\(UUID().uuidString).jpg")
        #if canImport(UIKit)
        guard let data = image.jpegData(compressionQuality: 0.9) else { return nil }
        #else
        guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil),
              let data = NSBitmapImageRep(cgImage: cgImage)
                  .representation(using: .jpeg, properties: [:]) else { return nil }
        #endif
        try? data.write(to: url)
        return url
    }
}

// MARK: - Platform image typealias

#if canImport(UIKit)
public typealias PlatformImage = UIImage
#elseif canImport(AppKit)
public typealias PlatformImage = NSImage
#endif

extension SwiftUI.Image {
    public init(platformImage: PlatformImage) {
#if canImport(UIKit)
        self.init(uiImage: platformImage)
#elseif canImport(AppKit)
        self.init(nsImage: platformImage)
#endif
    }
}

extension SwiftUI.Color {
    public static var outputBackground: Color {
        #if canImport(UIKit)
        return Color(.systemGroupedBackground)
        #elseif canImport(AppKit)
        return Color(nsColor: .windowBackgroundColor)
        #else
        return Color.secondary.opacity(0.1)
        #endif
    }
}
