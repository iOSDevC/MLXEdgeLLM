import Foundation

// MARK: - Model

/// A supported on-device model available for download and inference via MLX.
///
/// Each case maps to a Hugging Face repository (the raw value) and carries
/// metadata such as ``purpose``, ``displayName``, ``approximateSizeMB``,
/// and ``defaultDocumentPrompt``.
///
/// Models are grouped into three categories:
/// - **Text** — language-only generation (e.g. ``qwen3_1_7b``).
/// - **Vision** — multimodal image+text (e.g. ``qwen35_0_8b``).
/// - **Vision Specialized** — OCR / document extraction (e.g. ``fastVLM_0_5b_fp16``).
///
/// Use the convenience collections ``textModels``, ``visionModels``, and
/// ``specializedModels`` to list available models by category.
public enum Model: String, CaseIterable, Sendable {

    // MARK: Text
    case qwen3_0_6b   = "mlx-community/Qwen3-0.6B-4bit"
    case qwen3_1_7b   = "mlx-community/Qwen3-1.7B-4bit"
    case qwen3_4b     = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    case gemma3_1b    = "mlx-community/gemma-3-1b-it-4bit"
    case phi3_5_mini  = "mlx-community/Phi-3.5-mini-instruct-4bit"
    case llama3_2_1b  = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    case llama3_2_3b  = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    
    // MARK: Vision
    case qwen35_0_8b  = "mlx-community/Qwen3.5-0.8B-MLX-4bit"
    case qwen35_2b    = "mlx-community/Qwen3.5-2B-4bit"
    case smolvlm_500m = "mlx-community/SmolVLM-500M-Instruct-bf16"
    case smolvlm_2b   = "mlx-community/SmolVLM-Instruct-4bit"
    
    // MARK: Vision · Specialized
    case fastVLM_0_5b_fp16   = "mlx-community/FastVLM-0.5B-bf16"
    case fastVLM_1_5b_int8   = "InsightKeeper/FastVLM-1.5B-MLX-8bit"
    case graniteDocling_258m  = "ibm-granite/granite-docling-258M-mlx"
    case graniteVision_3_3   = "mlx-community/granite-vision-3.2-2b-MLX"
    
    // MARK: - Purpose

    /// The functional category of a model, determining which ``MLXEdgeLLM``
    /// factory method should be used to load it.
    public enum Purpose {
        /// Language-only generation — load with ``MLXEdgeLLM/text(_:onProgress:)``.
        case text
        /// Multimodal image+text — load with ``MLXEdgeLLM/vision(_:onProgress:)``.
        case vision
        /// OCR / document extraction — load with ``MLXEdgeLLM/specialized(_:onProgress:)``.
        /// When `docTags` is `true` the model outputs DocTags markup (e.g. Granite Docling).
        case visionSpecialized(docTags: Bool)
    }
    
    /// The functional category of this model.
    public var purpose: Purpose {
        switch self {
            case .qwen3_0_6b, .qwen3_1_7b, .qwen3_4b,
                    .gemma3_1b, .phi3_5_mini,
                    .llama3_2_1b, .llama3_2_3b:
                return .text
                
            case .qwen35_0_8b, .qwen35_2b,
                    .smolvlm_500m, .smolvlm_2b:
                return .vision
                
            case .fastVLM_0_5b_fp16, .fastVLM_1_5b_int8, .graniteVision_3_3:
                return .visionSpecialized(docTags: false)
                
            case .graniteDocling_258m:
                return .visionSpecialized(docTags: true)
        }
    }
    
    // MARK: - Metadata
    
    /// A human-readable name for display in UI (e.g. "Qwen3 1.7B").
    public var displayName: String {
        switch self {
            case .qwen3_0_6b:          return "Qwen3 0.6B"
            case .qwen3_1_7b:          return "Qwen3 1.7B"
            case .qwen3_4b:            return "Qwen3 4B"
            case .gemma3_1b:           return "Gemma 3 1B"
            case .phi3_5_mini:         return "Phi-3.5 Mini"
            case .llama3_2_1b:         return "Llama 3.2 1B"
            case .llama3_2_3b:         return "Llama 3.2 3B"
            case .qwen35_0_8b:         return "Qwen3.5 0.8B (default)"
            case .qwen35_2b:           return "Qwen3.5 2B"
            case .smolvlm_500m:        return "SmolVLM 500M"
            case .smolvlm_2b:          return "SmolVLM2 2B"
            case .fastVLM_0_5b_fp16:   return "FastVLM 0.5B FP16"
            case .fastVLM_1_5b_int8:   return "FastVLM 1.5B Int8"
            case .graniteDocling_258m: return "Granite Docling 258M (IBM)"
            case .graniteVision_3_3:   return "Granite Vision 3.2 2B"
        }
    }
    
    /// Approximate download size in megabytes. Use this to display storage
    /// requirements before the user downloads a model.
    public var approximateSizeMB: Int {
        switch self {
            case .qwen3_0_6b:          return 400
            case .qwen3_1_7b:          return 1_000
            case .qwen3_4b:            return 2_500
            case .gemma3_1b:           return 700
            case .phi3_5_mini:         return 2_200
            case .llama3_2_1b:         return 700
            case .llama3_2_3b:         return 1_800
            case .qwen35_0_8b:         return 625
            case .qwen35_2b:           return 1_720
            case .smolvlm_500m:        return 1_020
            case .smolvlm_2b:          return 1_460
            case .fastVLM_0_5b_fp16:   return 1_250
            case .fastVLM_1_5b_int8:   return 800
            case .graniteDocling_258m: return 631
            case .graniteVision_3_3:   return 1_200
        }
    }
    
    /// Default prompt for document/OCR extraction.
    /// Returns `nil` for non-specialized models.
    public var defaultDocumentPrompt: String? {
        switch self {
            case .fastVLM_0_5b_fp16, .fastVLM_1_5b_int8:
                return """
            You are a receipt OCR assistant. Extract all information from this receipt image \
            and return a JSON object with keys: store, date (YYYY-MM-DD), \
            items (array of {name, quantity, price}), subtotal, tax, total, currency. \
            Respond ONLY with valid JSON, no markdown.
            """
            case .graniteDocling_258m:
                return "Convert this page to docling."
            case .graniteVision_3_3:
                return "Describe the image in detail."
            default:
                return nil
        }
    }
    
    // MARK: - Storage
    
    /// Local cache directory where MLX downloads the model.
    /// Mirrors the `<org>/<repo>` folder structure used by mlx-swift.
    public var cacheDirectory: URL {
        FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("models")
            .appendingPathComponent(rawValue)
    }
    
    /// Returns `true` if the model directory exists on disk.
    public var isDownloaded: Bool {
        var isDir: ObjCBool = false
        let exists = FileManager.default.fileExists(
            atPath: cacheDirectory.path,
            isDirectory: &isDir
        )
        return exists && isDir.boolValue
    }
}

// MARK: - Convenience collections

public extension Model {
    /// All text-generation models, downloaded first.
    static var textModels: [Model] {
        allCases
            .filter { if case .text = $0.purpose { true } else { false } }
            .sorted { $0.isDownloaded && !$1.isDownloaded }
    }
    /// All general-purpose vision models, downloaded first.
    static var visionModels: [Model] {
        allCases
            .filter { if case .vision = $0.purpose { true } else { false } }
            .sorted { $0.isDownloaded && !$1.isDownloaded }
    }
    /// All OCR / document-specialized vision models, downloaded first.
    static var specializedModels: [Model] {
        allCases
            .filter { if case .visionSpecialized = $0.purpose { true } else { false } }
            .sorted { $0.isDownloaded && !$1.isDownloaded }
    }
}
