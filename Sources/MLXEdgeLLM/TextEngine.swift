import Foundation
import MLX
import MLXLLM
import MLXLMCommon

// MARK: - TextEngine

/// Internal class (NOT actor) that wraps ModelContainer for text LLM inference.
/// Uses @MainActor on the class so it's safe to call from SwiftUI, but the
/// heavy inference runs inside modelContainer.perform { } which hops off MainActor.
@MainActor
final class TextEngine {

    // MARK: - State

    private var modelContainer: ModelContainer?
    private let model: TextModel
    private let generateParameters: GenerateParameters

    // MARK: - Init

    init(model: TextModel, temperature: Float = 0.7, maxTokens: Int = 1024) {
        self.model = model
        self.generateParameters = GenerateParameters(temperature: temperature)
    }

    // MARK: - Load

    func load(onProgress: @escaping (String) -> Void) async throws {
        guard modelContainer == nil else { return }

        MLX.GPU.set(cacheLimit: 32 * 1024 * 1024)

        let config = ModelConfiguration(id: model.rawValue)
        modelContainer = try await LLMModelFactory.shared.loadContainer(
            configuration: config
        ) { progress in
            let pct = Int(progress.fractionCompleted * 100)
            Task { @MainActor in
                onProgress("Downloading \(progress.fileCompletedCount)/\(progress.fileTotalCount) — \(pct)%")
            }
        }
        onProgress("Model ready.")
    }

    // MARK: - Generate (full response, streaming via callback)

    /// Generate a response, calling `onToken` with each partial text as it streams.
    /// Returns the final complete text when done.
    func generate(
        prompt: String,
        systemPrompt: String?,
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

        return try await container.perform { context in
            let input = try await context.processor.prepare(
                input: .init(messages: messages)
            )
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
}
