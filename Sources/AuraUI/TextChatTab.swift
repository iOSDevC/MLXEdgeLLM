import SwiftUI
import AuraCore

// MARK: - Text Chat Tab
public struct TextChatTab: View {
    @StateObject private var vm = TextChatViewModel()
    @State private var selectedModel: Model = .qwen3_1_7b
    @State private var showConversations = false
    
    public init() {}
    
    public var body: some View {
        NavigationStack {
            ChatView(vm: vm, selectedModel: $selectedModel)
                .navigationTitle(vm.activeConversation?.title ?? "Text Chat")
#if os(iOS)
                .navigationBarTitleDisplayMode(.inline)
#endif
                .toolbar {
                    ToolbarItem(placement: .navigation) {
                        Button { showConversations = true } label: {
                            Image(systemName: "sidebar.left")
                        }
                    }
                    ToolbarItem(placement: .primaryAction) {
                        Menu {
                            ForEach(Model.textModels, id: \.self) { m in
                                Button {
                                    selectedModel = m
                                    Task { await vm.newConversation(model: m) }
                                } label: {
                                    HStack {
                                        Text(m.displayName)
                                        if m.isDownloaded {
                                            Image(systemName: "checkmark.circle.dotted")
                                        } else {
                                            Image(systemName: "square.and.arrow.down")
                                        }
                                    }
                                }
                            }
                        } label: {
                            Image(systemName: "square.and.pencil")
                        }
                    }
                }
                .sheet(isPresented: $showConversations) {
                    ConversationListSheet(vm: vm, isPresented: $showConversations)
                }
                .onTapGesture {
                    hidekeyboard()
                }
        }
    }
}

@inline(__always)
private func hidekeyboard() {
#if os(iOS)
    UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
#endif
}

#Preview {
    TextChatTab()
}
