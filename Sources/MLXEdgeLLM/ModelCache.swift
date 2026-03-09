import Foundation

// MARK: - ModelCache
//
// Singleton that loads each MLXEdgeLLM instance once and reuses it
// across all tabs. Prevents iOS from running out of RAM when multiple
// ViewModels try to load the same model independently.
//
// Usage:
//   let llm = try await ModelCache.shared.text(.qwen3_1_7b)
//   let vlm = try await ModelCache.shared.specialized(.fastVLM_0_5b_fp16)

@globalActor
public actor ModelCache {
    
    public static let shared = ModelCache()
    
    // MARK: - Cache storage
    
    private var textModels:       [Model: MLXEdgeLLM] = [:]
    private var visionModels:     [Model: MLXEdgeLLM] = [:]
    private var specializedModels:[Model: MLXEdgeLLM] = [:]
    
    // In-flight tasks — prevents duplicate concurrent loads of the same model
    private var pendingText:       [Model: Task<MLXEdgeLLM, Error>] = [:]
    private var pendingVision:     [Model: Task<MLXEdgeLLM, Error>] = [:]
    private var pendingSpecialized:[Model: Task<MLXEdgeLLM, Error>] = [:]
    
    private init() {}
    
    // MARK: - Public API
    
    /// Returns a cached text LLM, loading it on first call.
    public func text(
        _ model: Model = .qwen3_1_7b,
        onProgress: @escaping @MainActor (String) -> Void = { _ in }
    ) async throws -> MLXEdgeLLM {
        if let cached = textModels[model] { return cached }
        
        // Reuse in-flight task if already loading
        if let pending = pendingText[model] {
            return try await pending.value
        }
        
        let task = Task<MLXEdgeLLM, Error> {
            let llm = try await MLXEdgeLLM.text(model, onProgress: onProgress)
            textModels[model] = llm
            pendingText[model] = nil
            return llm
        }
        pendingText[model] = task
        return try await task.value
    }
    
    /// Returns a cached vision VLM, loading it on first call.
    public func vision(
        _ model: Model = .qwen35_0_8b,
        onProgress: @escaping @MainActor (String) -> Void = { _ in }
    ) async throws -> MLXEdgeLLM {
        if let cached = visionModels[model] { return cached }
        
        if let pending = pendingVision[model] {
            return try await pending.value
        }
        
        let task = Task<MLXEdgeLLM, Error> {
            let vlm = try await MLXEdgeLLM.vision(model, onProgress: onProgress)
            visionModels[model] = vlm
            pendingVision[model] = nil
            return vlm
        }
        pendingVision[model] = task
        return try await task.value
    }
    
    /// Returns a cached specialized VLM, loading it on first call.
    public func specialized(
        _ model: Model = .fastVLM_0_5b_fp16,
        onProgress: @escaping @MainActor (String) -> Void = { _ in }
    ) async throws -> MLXEdgeLLM {
        if let cached = specializedModels[model] { return cached }
        
        if let pending = pendingSpecialized[model] {
            return try await pending.value
        }
        
        let task = Task<MLXEdgeLLM, Error> {
            let vlm = try await MLXEdgeLLM.specialized(model, onProgress: onProgress)
            specializedModels[model] = vlm
            pendingSpecialized[model] = nil
            return vlm
        }
        pendingSpecialized[model] = task
        return try await task.value
    }
    
    // MARK: - Cache management
    
    /// Returns true if the model is already loaded in memory.
    public func isLoaded(_ model: Model) -> Bool {
        textModels[model] != nil ||
        visionModels[model] != nil ||
        specializedModels[model] != nil
    }
    
    /// Evicts a specific model from cache to free memory.
    public func evict(_ model: Model) {
        textModels[model]        = nil
        visionModels[model]      = nil
        specializedModels[model] = nil
    }
    
    /// Evicts all cached models — call on memory warning.
    public func evictAll() {
        textModels        = [:]
        visionModels      = [:]
        specializedModels = [:]
    }
}
