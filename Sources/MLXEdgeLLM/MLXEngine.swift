import Foundation
import MLX
import MLXLLM
import MLXVLM
import MLXLMCommon

#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

// MARK: - MLXEngine

/// Unified internal engine that dispatches to LLMModelFactory or VLMModelFactory
/// based on the model's purpose.
@MainActor
public final class MLXEngine {

    // MARK: - State

    private var modelContainer: ModelContainer?
    private let model: Model
    private let generateParameters: GenerateParameters

    // MARK: - Init

    public init(model: Model, temperature: Float? = nil) {
        self.model = model
        let defaultTemp: Float
        switch model.purpose {
            case .text:               defaultTemp = 0.7
            case .vision:             defaultTemp = 0.1
            case .visionSpecialized:  defaultTemp = 0.0   // deterministic OCR
        }
        self.generateParameters = GenerateParameters(temperature: temperature ?? defaultTemp)
    }

    // MARK: - Load

    func load(onProgress: @escaping (String) -> Void) async throws {
        guard modelContainer == nil else { return }

        let cacheLimitBytes: Int
        switch model.purpose {
            case .text:               cacheLimitBytes = 32 * 1024 * 1024
            case .vision:             cacheLimitBytes = 64 * 1024 * 1024
            case .visionSpecialized:  cacheLimitBytes = 48 * 1024 * 1024
        }
        MLX.GPU.set(cacheLimit: cacheLimitBytes)

        let config = ModelConfiguration(id: model.rawValue)

        switch model.purpose {
            case .text:
                modelContainer = try await LLMModelFactory.shared.loadContainer(
                    configuration: config
                ) { [model] progress in
                    let pct = Int(progress.fractionCompleted * 100)
                    Task { @MainActor in
                        onProgress("Downloading \(model.displayName): \(pct)%")
                    }
                }

            case .vision, .visionSpecialized:
                modelContainer = try await VLMModelFactory.shared.loadContainer(
                    configuration: config
                ) { [model] progress in
                    let pct = Int(progress.fractionCompleted * 100)
                    Task { @MainActor in
                        onProgress("Downloading \(model.displayName): \(pct)%")
                    }
                }
        }

        onProgress("\(model.displayName) ready ✓")
    }

    // MARK: - Generate (text-only)

    func generate(
        prompt: String,
        systemPrompt: String? = nil,
        maxTokens: Int = 1024,
        onToken: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        guard let container = modelContainer else {
            throw MLXEdgeLLMError.modelNotLoaded
        }

        var messages: [[String: String]] = []
        if let sys = systemPrompt {
            messages.append(["role": "system", "content": sys])
        }
        messages.append(["role": "user", "content": prompt])

        return try await performGeneration(
            container: container,
            prepareInput: { context in
                try await context.processor.prepare(input: .init(messages: messages))
            },
            maxTokens: maxTokens,
            onToken: onToken
        )
    }

    // MARK: - Generate (vision)

    func generate(
        prompt: String,
        image: PlatformImage?,
        maxTokens: Int = 800,
        onToken: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        guard let container = modelContainer else {
            throw MLXEdgeLLMError.modelNotLoaded
        }

        var tempURL: URL?
        let userInput: UserInput
        if let img = image, let url = saveImageToTemp(img) {
            tempURL = url
            userInput = UserInput(prompt: prompt, images: [.url(url)])
        } else {
            userInput = UserInput(prompt: prompt)
        }

        // Clean up temp JPEG after generation completes (fixes Finding 2.1)
        defer {
            if let url = tempURL {
                try? FileManager.default.removeItem(at: url)
            }
        }

        return try await performGeneration(
            container: container,
            prepareInput: { context in
                try await context.processor.prepare(input: userInput)
            },
            maxTokens: maxTokens,
            onToken: onToken
        )
    }

    // MARK: - Generate (multi-turn messages)

    /// Run inference with a pre-built messages array, preserving full multi-turn context.
    func generate(
        messages: [[String: String]],
        maxTokens: Int = 1024,
        onToken: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        guard let container = modelContainer else {
            throw MLXEdgeLLMError.modelNotLoaded
        }

        return try await performGeneration(
            container: container,
            prepareInput: { context in
                try await context.processor.prepare(input: .init(messages: messages))
            },
            maxTokens: maxTokens,
            onToken: onToken
        )
    }

    // MARK: - Core generation

    /// Shared generation logic. Uses a cancellation-aware token delivery pattern:
    /// - Checks `Task.isCancelled` before each MainActor hop to prevent
    ///   orphaned Tasks from updating deallocated ViewModels.
    /// - Returns `.stop` when the parent Task is cancelled, terminating
    ///   generation early instead of producing tokens nobody will read.
    private func performGeneration(
        container: ModelContainer,
        prepareInput: @escaping @Sendable (ModelContext) async throws -> LMInput,
        maxTokens: Int,
        onToken: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        return try await container.perform { context in
            let input = try await prepareInput(context)
            let result = try MLXLMCommon.generate(
                input: input,
                parameters: self.generateParameters,
                context: context
            ) { tokens in
                // Stop generating if the parent Task was cancelled
                // (e.g. user navigated away or tapped stop)
                guard !Task.isCancelled else { return .stop }

                let partial = context.tokenizer.decode(tokens: tokens)
                Task { @MainActor in
                    // Double-check cancellation before delivering to UI
                    guard !Task.isCancelled else { return }
                    onToken(partial)
                }
                return tokens.count >= maxTokens ? .stop : .more
            }
            return context.tokenizer.decode(tokens: result.tokens)
        }
    }

    // MARK: - Helpers

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
