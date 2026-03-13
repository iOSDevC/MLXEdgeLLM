import Foundation
import Dispatch
import MLXEdgeLLM
#if os(iOS) || os(tvOS)
import UIKit
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
// - Deduplicates concurrent loads of the same model via in-flight Task map.
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

    /// In-flight load tasks — deduplicates concurrent requests for the same model.
    private var inFlight: [Model: Task<MLXEdgeLLM, Error>] = [:]

    /// Max models to keep in RAM simultaneously (adaptive, based on device RAM).
    public private(set) var memoryBudget: Int

    private init() {
        self.memoryBudget = Self.detectMemoryBudget()
        observeMemoryPressure()
    }

    // MARK: - Memory pressure

    /// Registers for OS memory warnings and evicts LRU models to free RAM.
    private func observeMemoryPressure() {
        #if os(iOS) || os(tvOS)
        NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.handleMemoryPressure()
            }
        }
        #endif

        // DispatchSource memory pressure works on all Apple platforms including macOS
        let source = DispatchSource.makeMemoryPressureSource(
            eventMask: [.warning, .critical],
            queue: .main
        )
        source.setEventHandler { [weak self] in
            Task { @MainActor [weak self] in
                self?.handleMemoryPressure()
            }
        }
        source.resume()
        // Hold a reference so the source stays alive
        memoryPressureSource = source
    }

    private var memoryPressureSource: Any?

    private func handleMemoryPressure() {
        // Evict LRU models until only the most recently used remains
        while cache.count > 1, let lru = lruOrder.last {
            evict(lru)
        }
    }

    // MARK: - Public API

    /// Load a model and return the instance.
    /// Returns immediately from cache if already loaded.
    /// Deduplicates concurrent loads — multiple callers requesting the same model
    /// share a single in-flight load Task.
    public func load(
        _ model: Model,
        onProgress: (@MainActor (String) -> Void)? = nil
    ) async throws -> MLXEdgeLLM {
        // Return cached immediately
        if let cached = cache[model] {
            touch(model)
            return cached
        }

        // Deduplicate: if this model is already loading, join the existing Task
        if let existing = inFlight[model] {
            return try await existing.value
        }

        // Create a new load Task and register it
        let task = Task<MLXEdgeLLM, Error> { @MainActor [weak self] in
            guard let self else { throw MLXEdgeLLMError.modelNotLoaded }
            defer { self.inFlight[model] = nil }
            return try await self.performLoad(model, onProgress: onProgress)
        }
        inFlight[model] = task
        return try await task.value
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

    /// Whether the model is already loaded in memory.
    /// Alias for compatibility with code previously using ModelCache.isLoaded().
    public func isLoaded(_ model: Model) -> Bool {
        cache[model] != nil
    }

    // MARK: - Load

    private func performLoad(
        _ model: Model,
        onProgress: (@MainActor (String) -> Void)? = nil
    ) async throws -> MLXEdgeLLM {
        states[model] = .loading

        // Evict LRU if over budget
        while cache.count >= memoryBudget, let lru = lruOrder.last {
            evict(lru)
        }

        let progress: @MainActor (String) -> Void = { [weak self] p in
            self?.states[model] = .downloading(progress: p)
            onProgress?(p)
        }

        let instance: MLXEdgeLLM
        switch model.purpose {
            case .text:
                instance = try await MLXEdgeLLM.text(model, onProgress: progress)
            case .vision:
                instance = try await MLXEdgeLLM.vision(model, onProgress: progress)
            case .visionSpecialized:
                instance = try await MLXEdgeLLM.specialized(model, onProgress: progress)
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
