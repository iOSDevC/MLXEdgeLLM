import Foundation

// MARK: - MLXEdgeLLM  (public text API)

/// On-device text LLM powered by mlx-swift-lm.
///
/// ```swift
/// // One-liner
/// let reply = try await MLXEdgeLLM.chat("¿Cuánto gasté esta semana?")
///
/// // Reusable instance with streaming
/// let llm = try await MLXEdgeLLM(model: .qwen3_1_7b)
/// for try await token in llm.stream("Describe my expenses") {
///     print(token, terminator: "")
/// }
/// ```
@MainActor
public final class MLXEdgeLLM {

    // MARK: - Properties

    private let engine: TextEngine
    public let model: TextModel

    // MARK: - Init

    /// Load a text model. Downloads on first use, then cached locally.
    public init(
        model: TextModel = .qwen3_1_7b,
        onProgress: @escaping (String) -> Void = { _ in }
    ) async throws {
        self.model = model
        self.engine = TextEngine(model: model)
        try await engine.load(onProgress: onProgress)
    }

    // MARK: - Chat

    /// Send a message and get the full response.
    public func chat(
        _ prompt: String,
        systemPrompt: String? = nil,
        maxTokens: Int = 1024
    ) async throws -> String {
        try await engine.generate(
            prompt: prompt,
            systemPrompt: systemPrompt,
            maxTokens: maxTokens,
            onToken: { _ in }
        )
    }

    /// Stream tokens one by one as an `AsyncThrowingStream`.
    public func stream(
        _ prompt: String,
        systemPrompt: String? = nil,
        maxTokens: Int = 1024
    ) -> AsyncThrowingStream<String, Error> {
        // Capture self (MainActor-isolated) into a detached Task safely
        let engine = self.engine
        let sys = systemPrompt

        return AsyncThrowingStream { continuation in
            Task { @MainActor in
                do {
                    var lastLength = 0
                    _ = try await engine.generate(
                        prompt: prompt,
                        systemPrompt: sys,
                        maxTokens: maxTokens
                    ) { @MainActor partial in
                        // Yield only the NEW characters since last callback
                        let newText = String(partial.dropFirst(lastLength))
                        lastLength = partial.count
                        if !newText.isEmpty {
                            continuation.yield(newText)
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // MARK: - Static convenience

    /// One-liner text chat (loads model fresh each call — prefer instance for reuse).
    public static func chat(
        _ prompt: String,
        model: TextModel = .qwen3_1_7b,
        systemPrompt: String? = nil,
        onProgress: @escaping (String) -> Void = { _ in }
    ) async throws -> String {
        let llm = try await MLXEdgeLLM(model: model, onProgress: onProgress)
        return try await llm.chat(prompt, systemPrompt: systemPrompt)
    }
}
