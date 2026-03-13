import SwiftUI
import AuraCore

struct RunButton: View {
    let title: String
    let subtitle: String
    let isDownloaded: Bool
    let isLoading: Bool
    let color: Color
    let action: () async -> Void
    
    var body: some View {
        Button {
            Task { await action() }
        } label: {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(.headline)
                    Text(subtitle)
                        .font(.caption)
                        .opacity(0.8)
                }
                Spacer()
                Group {
                    if isLoading {
                        ProgressView()
                            .tint(.white)
                    } else if isDownloaded {
                        Image(systemName: "play.fill")
                    } else {
                        Image(systemName: "arrow.down.circle")
                    }
                }
                .font(.title3)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .background(color, in: RoundedRectangle(cornerRadius: 12))
            .foregroundStyle(.white)
        }
        .buttonStyle(.plain)
        .disabled(isLoading)
    }
}
