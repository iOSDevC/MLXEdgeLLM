import SwiftUI
import PhotosUI
import AuraCore
import AuraUI
import AuraVoice
import AuraDocs
import AuraAppleIntelligence

// MARK: - ContentView

struct ContentView: View {
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        TabView {
            TextChatTab()
                .tabItem { Label("Text", systemImage: "text.bubble") }
            
            VoiceTab()
                .tabItem { Label("Voice", systemImage: "mic.fill") }
            
            DocsTab()
                .tabItem { Label("Docs", systemImage: "doc.text.magnifyingglass") }
            
            VisionTab()
                .tabItem { Label("Vision", systemImage: "eye") }
            
            OCRTab()
                .tabItem { Label("OCR", systemImage: "doc.viewfinder") }
            
            ModelsTab()
                .tabItem { Label("Models", systemImage: "square.stack.3d.up") }
            
            // AgentCrew — requires iOS 26+ with Apple Intelligence
            if #available(iOS 26, macOS 26, *) {
                AgentCrewMainView()
                    .environmentObject(appState)
                    .tabItem { Label("Agents", systemImage: "cpu") }
            }
        }
    }
}

// MARK: - Models Tab

struct ModelsTab: View {
    var body: some View {
        NavigationStack {
            List {
                ModelSection(title: "Text",           icon: "text.bubble",    color: .green,  models: Model.textModels)
                ModelSection(title: "Vision",         icon: "eye",            color: .blue,   models: Model.visionModels)
                ModelSection(title: "Specialized OCR",icon: "doc.viewfinder", color: .orange, models: Model.specializedModels)
            }
            .navigationTitle("Models")
        }
    }
}

#Preview { ContentView().environmentObject(AppState()) }
