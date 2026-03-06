import Foundation

// MARK: - TextModel

public enum TextModel: String, CaseIterable, Sendable {
    case qwen3_0_6b   = "mlx-community/Qwen3-0.6B-Instruct-4bit"
    case qwen3_1_7b   = "mlx-community/Qwen3-1.7B-Instruct-4bit"
    case qwen3_4b     = "mlx-community/Qwen3-4B-Instruct-4bit"
    case gemma3_1b    = "mlx-community/gemma-3-1b-it-4bit"
    case phi3_5_mini  = "mlx-community/Phi-3.5-mini-instruct-4bit"
    case llama3_2_1b  = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    case llama3_2_3b  = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    
    public var displayName: String {
        switch self {
            case .qwen3_0_6b:  return "Qwen3 0.6B"
            case .qwen3_1_7b:  return "Qwen3 1.7B"
            case .qwen3_4b:    return "Qwen3 4B"
            case .gemma3_1b:   return "Gemma 3 1B"
            case .phi3_5_mini: return "Phi-3.5 Mini"
            case .llama3_2_1b: return "Llama 3.2 1B"
            case .llama3_2_3b: return "Llama 3.2 3B"
        }
    }
    public var approximateSizeMB: Int {
        switch self {
            case .qwen3_0_6b: return 400; case .qwen3_1_7b: return 1_000
            case .qwen3_4b: return 2_500; case .gemma3_1b: return 700
            case .phi3_5_mini: return 2_200; case .llama3_2_1b: return 700
            case .llama3_2_3b: return 1_800
        }
    }
}

// MARK: - VisionModel (general-purpose, MLXVLM backend)

public enum VisionModel: String, CaseIterable, Sendable {
    case qwen35_0_8b  = "mlx-community/Qwen3.5-0.8B-MLX-4bit"
    case qwen35_2b    = "mlx-community/Qwen3.5-2B-MLX-4bit"
    case qwen35_4b    = "mlx-community/Qwen3.5-4B-MLX-4bit"
    case qwen25vl_2b  = "mlx-community/Qwen2.5-VL-2B-Instruct-4bit"
    case gemma3_4b    = "mlx-community/gemma-3-4b-it-4bit"
    case smolvlm_500m = "mlx-community/SmolVLM-500M-Instruct-bf16"
    case smolvlm_2b   = "mlx-community/SmolVLM2-2.2B-Instruct-4bit"
    
    public var displayName: String {
        switch self {
            case .qwen35_0_8b:  return "Qwen3.5 0.8B (default)"
            case .qwen35_2b:    return "Qwen3.5 2B"
            case .qwen35_4b:    return "Qwen3.5 4B"
            case .qwen25vl_2b:  return "Qwen2.5-VL 2B"
            case .gemma3_4b:    return "Gemma 3 4B"
            case .smolvlm_500m: return "SmolVLM 500M"
            case .smolvlm_2b:   return "SmolVLM2 2B"
        }
    }
    public var approximateSizeMB: Int {
        switch self {
            case .qwen35_0_8b: return 1_000; case .qwen35_2b: return 1_800
            case .qwen35_4b: return 3_200; case .qwen25vl_2b: return 1_400
            case .gemma3_4b: return 2_500; case .smolvlm_500m: return 500
            case .smolvlm_2b: return 1_200
        }
    }
}

// MARK: - SpecializedVisionModel (OCR-optimized, MLXVLM backend)

/// Ultra-lightweight models specialized for OCR and document parsing.
///
/// **FastVLM 0.5B** — Apple CVPR 2025. 85× faster TTFT than LLaVA-0.5B.
/// HF: `apple/FastVLM-0.5B-fp16` (~420 MB, Sep 2025)
///
/// **Granite Docling 258M** — IBM Research. DocTags output preserving tables/equations.
/// HF: `ibm-granite/granite-docling-258M-mlx` (~270 MB, Sep 2025)
public enum SpecializedVisionModel: String, CaseIterable, Sendable {
    case fastVLM_0_6b_int8 = "apple/FastVLM-0.6B-int8"   // ~330 MB ← new default
    case fastVLM_0_5b_fp16 = "apple/FastVLM-0.5B-fp16"   // ~420 MB
    case fastVLM_1_5b_int8 = "apple/FastVLM-1.5B-int8"   // ~800 MB ← best OCR quality
    case graniteDocling_258m = "ibm-granite/granite-docling-258M-mlx"
    case graniteVision_3_3  = "mlx-community/granite-vision-3.2-2b-MLX"  // ~1.2 GB
    
    public var displayName: String {
        switch self {
            case .fastVLM_0_6b_int8: return "FastVLM 0.6B Int8 (Apple)"
            case .fastVLM_0_5b_fp16: return "FastVLM 0.5B FP16 (Apple)"
            case .fastVLM_1_5b_int8: return "FastVLM 1.5B Int8 (Apple)"
            case .graniteDocling_258m: return "Granite Docling 258M (IBM)"
            case .graniteVision_3_3: return "Granite Vision 3.2 2B"
        }
    }
    public var approximateSizeMB: Int {
        switch self {
            case .fastVLM_0_6b_int8: return 330
            case .fastVLM_0_5b_fp16: return 420
            case .fastVLM_1_5b_int8: return 800
            case .graniteDocling_258m: return 631
            case .graniteVision_3_3: return 1_200
        }
    }
    /// Default prompt for `extractDocument(_:)`.
    public var defaultDocumentPrompt: String {
        switch self {
            case .fastVLM_0_5b_fp16, .fastVLM_0_6b_int8, .fastVLM_1_5b_int8:
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
        }
    }
    /// Granite Docling outputs DocTags — use `MLXEdgeLLMSpecialized.parseDocTags(_:)`.
    public var outputsDocTags: Bool {
        self == .graniteDocling_258m
    }
}
