import Foundation
import MLXEdgeLLM
#if os(iOS) || os(tvOS) || os(watchOS)
import Darwin
#endif

// MARK: - ModelLoadState

/// Observable load state for a single model — used by UI to show progress.
public enum ModelLoadState: Equatable {
    case idle
    case downloading(progress: String)
    case loading
    case ready
    case failed(String)
    
    public var isActive: Bool {
        switch self {
            case .downloading, .loading: return true
            default: return false
        }
    }
}

// MARK: - ModelManager
//
// Single point of control for all MLX model loading across the app.
//
// - Detects available RAM and decides how many models to keep in memory (LRU).
// - Serializes concurrent load requests — only one model loads at a time.
// - Publishes per-model load state so any SwiftUI view can show progress.
//
// Usage:
//   let llm = try await ModelManager.shared.load(.qwen3_1_7b)
//   @ObservedObject var mm = ModelManager.shared  →  mm.state(for: model)

@MainActor
public final class ModelManager: ObservableObject {
    
    // MARK: - Singleton
    
    public static let shared = ModelManager()
    
    // MARK: - Published state
    
    @Published public private(set) var states: [Model: ModelLoadState] = [:]
    
    // MARK: - Private storage
    
    private var cache:    [Model: MLXEdgeLLM] = [:]
    private var lruOrder: [Model] = []
    
    private var pendingLoads: [(Model, CheckedContinuation<MLXEdgeLLM, Error>)] = []
    private var isLoading = false
    
    /// Max models to keep in RAM simultaneously (adaptive, based on device RAM).
    public private(set) var memoryBudget: Int
    
    private init() {
        self.memoryBudget = Self.detectMemoryBudget()
    }
    
    // MARK: - Public API
    
    /// Load a model and return the instance.
    /// Returns immediately from cache if already loaded.
    /// Queues and waits if another model is currently loading.
    public func load(
        _ model: Model,
        onProgress: (@MainActor (String) -> Void)? = nil
    ) async throws -> MLXEdgeLLM {
        if let cached = cache[model] {
            touch(model)
            return cached
        }
        return try await withCheckedThrowingContinuation { continuation in
            pendingLoads.append((model, continuation))
            Task { await self.drainQueue() }
        }
    }
    
    /// Current load state for a model.
    public func state(for model: Model) -> ModelLoadState {
        states[model] ?? .idle
    }
    
    /// Evict a specific model from memory.
    public func evict(_ model: Model) {
        cache[model] = nil
        lruOrder.removeAll { $0 == model }
        states[model] = .idle
    }
    
    /// Evict all loaded models.
    public func evictAll() {
        cache     = [:]
        lruOrder  = []
        for key in states.keys { states[key] = .idle }
    }
    
    /// Whether the model is currently loaded and ready.
    public func isReady(_ model: Model) -> Bool {
        cache[model] != nil
    }
    
    // MARK: - Queue
    
    private func drainQueue() async {
        guard !isLoading, !pendingLoads.isEmpty else { return }
        isLoading = true
        
        while !pendingLoads.isEmpty {
            let (model, continuation) = pendingLoads.removeFirst()
            
            // Re-check cache — another queued call may have loaded it already
            if let cached = cache[model] {
                touch(model)
                continuation.resume(returning: cached)
                continue
            }
            
            do {
                let instance = try await performLoad(model)
                continuation.resume(returning: instance)
            } catch {
                states[model] = .failed(error.localizedDescription)
                continuation.resume(throwing: error)
            }
        }
        
        isLoading = false
    }
    
    // MARK: - Load
    
    private func performLoad(_ model: Model) async throws -> MLXEdgeLLM {
        states[model] = .loading
        
        // Evict LRU if over budget
        while cache.count >= memoryBudget, let lru = lruOrder.last {
            evict(lru)
        }
        
        let onProgress: @MainActor (String) -> Void = { [weak self] p in
            self?.states[model] = .downloading(progress: p)
        }
        
        let instance: MLXEdgeLLM
        switch model.purpose {
            case .text:
                instance = try await MLXEdgeLLM.text(model, onProgress: onProgress)
            case .vision:
                instance = try await MLXEdgeLLM.vision(model, onProgress: onProgress)
            case .visionSpecialized:
                instance = try await MLXEdgeLLM.specialized(model, onProgress: onProgress)
        }
        
        cache[model] = instance
        touch(model)
        states[model] = .ready
        return instance
    }
    
    // MARK: - LRU
    
    private func touch(_ model: Model) {
        lruOrder.removeAll { $0 == model }
        lruOrder.insert(model, at: 0)
    }
    
    // MARK: - Memory detection
    
    private static func detectMemoryBudget() -> Int {
        let available = availableMemoryBytes()
        let usable    = max(0, available - 2 * 1024 * 1024 * 1024) // reserve 2 GB for OS
        let budget    = max(1, Int(usable / (1_500 * 1024 * 1024))) // ~1.5 GB per model
        return min(budget, 4)
    }
    
    private static func availableMemoryBytes() -> Int {
#if os(iOS) || os(tvOS) || os(watchOS)
        let available = os_proc_available_memory()
        if available > 0 { return Int(available) }
#endif
        return Int(Double(ProcessInfo.processInfo.physicalMemory) * 0.6)
    }
}

// MARK: - ModelLoadingOverlay (SwiftUI)

#if canImport(SwiftUI)
import SwiftUI

/// Drop-in overlay that shows model load progress.
/// Renders nothing when the model is idle or ready.
public struct ModelLoadingOverlay: View {
    let model: Model
    @ObservedObject private var manager = ModelManager.shared
    
    public init(model: Model) { self.model = model }
    
    public var body: some View {
        let s = manager.state(for: model)
        if s.isActive {
            VStack(spacing: 12) {
                ProgressView().scaleEffect(1.2)
                switch s {
                    case .downloading(let p):
                        Text(p)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                    case .loading:
                        Text("Loading \(model.displayName)…")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    default:
                        EmptyView()
                }
            }
            .padding(24)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
        }
    }
}
#endif
