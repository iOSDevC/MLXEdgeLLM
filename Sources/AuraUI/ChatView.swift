import SwiftUI
import AuraCore

// MARK: - Chat View
public struct ChatView: View {
    @ObservedObject var vm: TextChatViewModel
    @Binding var selectedModel: Model
    @State private var prompt: String = ""
    @FocusState private var promptFocused: Bool
    
    public var body: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        ForEach(vm.messages) { msg in
                            MessageBubble(message: msg).id(msg.id)
                        }
                        if vm.isStreaming {
                            StreamingBubble(text: vm.streamingText).id("streaming")
                        }
                    }
                    .padding()
                }
                .onChange(of: vm.streamingText) { _, _ in
                    withAnimation { proxy.scrollTo("streaming", anchor: .bottom) }
                }
                .onChange(of: vm.messages.count) { _, _ in
                    withAnimation { proxy.scrollTo(vm.messages.last?.id, anchor: .bottom) }
                }
            }
            
            Divider()
            
            if !vm.progress.isEmpty {
                HStack(spacing: 6) {
                    ProgressView().scaleEffect(0.7)
                    Text(vm.progress).font(.caption).foregroundStyle(.secondary)
                }
                .padding(.horizontal)
                .padding(.top, 6)
            }
            
            HStack(spacing: 10) {
                TextField("Ask something...", text: $prompt, axis: .vertical)
                    .lineLimit(1...4)
                    .padding(10)
                    .background(Color.tertiaryGroupedBackground, in: RoundedRectangle(cornerRadius: 12))
                    .focused($promptFocused)
                
                Button {
                    let text = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !text.isEmpty, !vm.isStreaming else { return }
                    prompt = ""
                    promptFocused = false
                    Task { await vm.send(text, model: selectedModel) }
                } label: {
                    Image(systemName: vm.isStreaming ? "stop.circle.fill" : "arrow.up.circle.fill")
                        .font(.system(size: 32))
                        .foregroundStyle(vm.isStreaming ? .red : .blue)
                }
                .disabled(prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !vm.isStreaming)
            }
            .padding(12)
            .background(Color.secondaryGroupedBackground)
        }
        .background(Color.groupedBackground)
    }
}
