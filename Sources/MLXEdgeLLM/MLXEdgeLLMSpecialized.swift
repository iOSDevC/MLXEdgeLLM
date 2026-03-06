import Foundation

#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

// MARK: - MLXEdgeLLMSpecialized

/// Public API for specialized OCR/document models (FastVLM 0.5B, Granite Docling 258M).
///
/// ```swift
/// // Receipt JSON — uses FastVLM 0.5B (fastest, ~420 MB)
/// let json = try await MLXEdgeLLMSpecialized.extractReceipt(ticketImage)
///
/// // Document → Markdown — uses Granite Docling 258M (~270 MB)
/// let engine = try await MLXEdgeLLMSpecialized(model: .graniteDocling_258m)
/// let docTags = try await engine.extractDocument(scanImage)
/// let markdown = MLXEdgeLLMSpecialized.parseDocTags(docTags)
///
/// // Streaming
/// for try await token in engine.stream("What is the total?", image: receipt) {
///     print(token, terminator: "")
/// }
/// ```
@MainActor
public final class MLXEdgeLLMSpecialized {

    // MARK: - Properties

    private let engine: SpecializedVisionEngine
    public let model: SpecializedVisionModel

    // MARK: - Init

    public init(
        model: SpecializedVisionModel = .fastVLM_0_5b_fp16,
        onProgress: @escaping (String) -> Void = { _ in }
    ) async throws {
        self.model = model
        self.engine = SpecializedVisionEngine(model: model)
        try await engine.load(onProgress: onProgress)
    }

    // MARK: - Receipt extraction

    /// Extract structured JSON from a receipt image.
    public func extractReceipt(_ image: PlatformImage) async throws -> String {
        try await engine.generate(
            prompt: model.defaultDocumentPrompt,
            image: image,
            maxTokens: 600,
            onToken: { _ in }
        )
    }

    /// One-liner: extract receipt JSON using FastVLM 0.5B.
    public static func extractReceipt(
        _ image: PlatformImage,
        onProgress: @escaping (String) -> Void = { _ in }
    ) async throws -> String {
        let e = try await MLXEdgeLLMSpecialized(model: .fastVLM_0_5b_fp16, onProgress: onProgress)
        return try await e.extractReceipt(image)
    }

    // MARK: - Document extraction

    /// Convert a document image using the model's default prompt.
    /// - FastVLM → JSON string
    /// - Granite Docling → DocTags string (call `parseDocTags(_:)` to get Markdown)
    public func extractDocument(_ image: PlatformImage) async throws -> String {
        try await engine.generate(
            prompt: model.defaultDocumentPrompt,
            image: image,
            maxTokens: model == .graniteDocling_258m ? 2048 : 600,
            onToken: { _ in }
        )
    }

    /// Ask a free-form question about an image (or text-only if image is nil).
    public func analyze(
        _ prompt: String,
        image: PlatformImage? = nil,
        maxTokens: Int = 512
    ) async throws -> String {
        try await engine.generate(
            prompt: prompt,
            image: image,
            maxTokens: maxTokens,
            onToken: { _ in }
        )
    }

    // MARK: - Streaming

    public func stream(
        _ prompt: String,
        image: PlatformImage? = nil,
        maxTokens: Int = 512
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

    // MARK: - DocTags → Markdown (Granite Docling post-processing)

    /// Convert Granite Docling's DocTags output to readable Markdown.
    ///
    /// For full-fidelity conversion (tables as HTML, formulas as LaTeX),
    /// pipe the DocTags string through Python's `docling-core` library.
    public static func parseDocTags(_ docTags: String) -> String {
        var md = docTags

        // Remove page/body wrappers
        for tag in ["<doctag>", "</doctag>", "<page>", "</page>", "<body>", "</body>"] {
            md = md.replacingOccurrences(of: tag, with: "")
        }

        // Headings
        for (tag, prefix) in [("section-header-1", "#"), ("section-header-2", "##"), ("section-header-3", "###")] {
            md = md.replacingOccurrences(
                of: "<\(tag)>(.*?)</\(tag)>",
                with: "\(prefix) $1",
                options: .regularExpression
            )
        }
        // Paragraphs
        md = md.replacingOccurrences(of: #"<paragraph>(.*?)</paragraph>"#, with: "$1\n", options: .regularExpression)
        // List items
        md = md.replacingOccurrences(of: #"<list-item>(.*?)</list-item>"#, with: "- $1", options: .regularExpression)
        // Formulas
        md = md.replacingOccurrences(of: #"<formula>(.*?)</formula>"#, with: "`$1`", options: .regularExpression)
        // Tables → code block placeholder
        md = md.replacingOccurrences(
            of: #"(?s)<table>(.*?)</table>"#,
            with: "\n```\n$1\n```\n",
            options: .regularExpression
        )

        return md
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
            .joined(separator: "\n\n")
    }
}
