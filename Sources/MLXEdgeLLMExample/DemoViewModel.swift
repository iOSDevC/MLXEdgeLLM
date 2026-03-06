//
//  File.swift
//  MLXEdgeLLM
//
//  Created by Cristopher Bautista on 3/6/26.
//

import Foundation
import UIKit
import MLXEdgeLLM

@MainActor
final class DemoViewModel: ObservableObject {
    @Published var output: String = ""
    @Published var progress: String = ""
    @Published var isLoading = false
    @Published var selectedImage: UIImage?
    
    func runVLM(model: VisionModel, image: UIImage) async {
        startLoading()
        do {
            let vlm = try await MLXEdgeLLMVision(model: model) { p in
                Task { @MainActor in self.progress = p }
            }
            let result = try await vlm.extractReceipt(image)
            output = "📋 \(model.displayName):\n\n\(result)"
        } catch { output = "❌ \(error.localizedDescription)" }
        stopLoading()
    }
    
    func runSpecialized(model: SpecializedVisionModel, image: UIImage) async {
        startLoading()
        do {
            let e = try await MLXEdgeLLMSpecialized(model: model) { p in
                Task { @MainActor in self.progress = p }
            }
            var result = try await e.extractDocument(image)
            if model.outputsDocTags {
                result = MLXEdgeLLMSpecialized.parseDocTags(result)
                output = "📝 Granite → Markdown:\n\n\(result)"
            } else {
                output = "⚡ FastVLM JSON:\n\n\(result)"
            }
        } catch { output = "❌ \(error.localizedDescription)" }
        stopLoading()
    }
    
    func runStreamVLM(model: VisionModel, image: UIImage) async {
        startLoading()
        output = ""
        do {
            let vlm = try await MLXEdgeLLMVision(model: model) { p in
                Task { @MainActor in self.progress = p }
            }
            for try await token in vlm.stream("Describe this receipt in detail.", image: image) {
                output += token
            }
        } catch { output = "❌ \(error.localizedDescription)" }
        stopLoading()
    }
    
    func runTextChat() async {
        startLoading()
        do {
            let llm = try await MLXEdgeLLM(model: .qwen3_1_7b) { p in
                Task { @MainActor in self.progress = p }
            }
            output = ""
            for try await token in llm.stream("¿Cuánto es el IVA en México y cómo aparece en tickets?") {
                output += token
            }
        } catch { output = "❌ \(error.localizedDescription)" }
        stopLoading()
    }
    
    private func startLoading() { isLoading = true; progress = "" }
    private func stopLoading() { isLoading = false; progress = "" }
}
