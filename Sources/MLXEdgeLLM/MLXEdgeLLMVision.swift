import Foundation

#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

// MARK: - MLXEdgeLLMVision  (public VLM API)

/// On-device Vision-Language Model powered by mlx-swift-lm / MLXVLM.
///
/// ```swift
/// // One-liner receipt extraction
/// let json = try await MLXEdgeLLMVision.extractReceipt(ticketImage)
///
/// // Reusable instance
/// let vlm = try await MLXEdgeLLMVision(model: .qwen35_0_8b)
/// let json = try await vlm.extractReceipt(ticket)
///
/// // Streaming with image
/// for try await token in vlm.stream("Describe this receipt", image: photo) {
///     print(token, terminator: "")
/// }
/// ```
@MainActor
public final class MLXEdgeLLMVision {

    // MARK: - Properties

    private let engine: VisionEngine
    public let model: VisionModel

    // MARK: - Init

    public init(
        model: VisionModel = .qwen35_0_8b,
        onProgress: @escaping (String) -> Void = { _ in }
    ) async throws {
        self.model = model
        self.engine = VisionEngine(model: model)
        try await engine.load(onProgress: onProgress)
    }

    // MARK: - Receipt Extraction

    private static let receiptPrompt = """
        You are a receipt OCR assistant. Extract all information from this receipt image \
        and return a JSON object with keys: store, date (YYYY-MM-DD), \
        items (array of {name, quantity, price}), subtotal, tax, total, currency. \
        Respond ONLY with valid JSON, no markdown fences.
        """

    /// Extract structured receipt data (JSON string) from a receipt image.
    public func extractReceipt(_ image: PlatformImage) async throws -> String {
        try await engine.generate(
            prompt: Self.receiptPrompt,
            image: image,
            maxTokens: 600,
            onToken: { _ in }
        )
    }

    /// One-liner receipt extraction using default model.
    public static func extractReceipt(
        _ image: PlatformImage,
        onProgress: @escaping (String) -> Void = { _ in }
    ) async throws -> String {
        let vlm = try await MLXEdgeLLMVision(onProgress: onProgress)
        return try await vlm.extractReceipt(image)
    }

    // MARK: - Generic analyze

    /// Ask a custom question about an image (or text-only if image is nil).
    public func analyze(
        _ prompt: String,
        image: PlatformImage? = nil,
        maxTokens: Int = 800
    ) async throws -> String {
        try await engine.generate(
            prompt: prompt,
            image: image,
            maxTokens: maxTokens,
            onToken: { _ in }
        )
    }

    // MARK: - Streaming

    /// Stream tokens for a vision + text query.
    public func stream(
        _ prompt: String,
        image: PlatformImage? = nil,
        maxTokens: Int = 800
    ) -> AsyncThrowingStream<String, Error> {
        let engine = self.engine
        let img = image

        return AsyncThrowingStream { continuation in
            Task { @MainActor in
                do {
                    var lastLength = 0
                    _ = try await engine.generate(
                        prompt: prompt,
                        image: img,
                        maxTokens: maxTokens
                    ) { @MainActor partial in
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
}
