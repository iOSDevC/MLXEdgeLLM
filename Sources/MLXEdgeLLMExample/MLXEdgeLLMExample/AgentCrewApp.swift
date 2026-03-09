import SwiftUI
import MLXEdgeLLMAppleIntelligence

// MARK: - AgentCrewMainView
//
// Three-tab UI for the AgentCrew pipeline.
// Embedded as a tab inside ContentView — AppState is injected via @EnvironmentObject.

@available(iOS 26, macOS 26, *)
struct AgentCrewMainView: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        TabView {
            PipelineTab()
                .environmentObject(appState)
                .tabItem { Label("Pipeline", systemImage: "cpu") }

            DocumentsTab()
                .environmentObject(appState)
                .tabItem { Label("Documents", systemImage: "doc.text.magnifyingglass") }

            HistoryTab()
                .environmentObject(appState)
                .tabItem { Label("History", systemImage: "clock.arrow.circlepath") }
        }
        .task { await appState.setup() }
    }
}
