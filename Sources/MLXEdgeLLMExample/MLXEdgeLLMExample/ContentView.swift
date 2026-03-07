import SwiftUI
import PhotosUI
import MLXEdgeLLM
import MLXEdgeLLMUI
import MLXEdgeLLMVoice
import MLXEdgeLLMDocs

// MARK: - ContentView

struct ContentView: View {
    var body: some View {
        TabView {
            TextChatTab()
                .tabItem {
                    Label("Text", systemImage: "text.bubble")
                }
            
            VoiceTab()
                .tabItem {
                    Label("Voice", systemImage: "mic.fill")
                }
            // Agregar en el TabView existente:
            DocsTab()
                .tabItem { Label("Docs", systemImage: "doc.text.magnifyingglass") }
            
            VisionTab()
                .tabItem {
                    Label("Vision", systemImage: "eye")
                }
            
            OCRTab()
                .tabItem {
                    Label("OCR", systemImage: "doc.viewfinder")
                }
            
            ModelsTab()
                .tabItem {
                    Label("Models", systemImage: "square.stack.3d.up")
                }
        }
    }
}

// MARK: - Models Tab

public struct ModelsTab: View {
    public init() {}
    
    public var body: some View {
        NavigationStack {
            List {
                ModelSection(title: "Text", icon: "text.bubble", color: .green, models: Model.textModels)
                ModelSection(title: "Vision", icon: "eye", color: .blue, models: Model.visionModels)
                ModelSection(title: "Specialized OCR", icon: "doc.viewfinder", color: .orange, models: Model.specializedModels)
            }
            .navigationTitle("Models")
        }
    }
}


#Preview { ContentView() }
