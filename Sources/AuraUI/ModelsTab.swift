import SwiftUI
import AuraCore

public struct ModelSection: View {
    let title: String
    let icon: String
    let color: Color
    let models: [Model]
    
    public init(title: String, icon: String, color: Color, models: [Model]) {
        self.title = title
        self.icon = icon
        self.color = color
        self.models = models
    }
    
    public var body: some View {
        Section {
            ForEach(models, id: \.self) { model in
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(model.displayName)
                            .font(.subheadline.weight(.medium))
                        Text(model.rawValue)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                    Spacer()
                    VStack(alignment: .trailing, spacing: 2) {
                        Text("\(model.approximateSizeMB) MB")
                            .font(.caption.weight(.medium))
                            .foregroundStyle(.secondary)
                        if model.isDownloaded {
                            Label("Ready", systemImage: "checkmark.circle.fill")
                                .font(.caption2)
                                .foregroundStyle(.green)
                        } else {
                            Label("Not downloaded", systemImage: "arrow.down.circle")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                .padding(.vertical, 2)
            }
        } header: {
            Label(title, systemImage: icon)
                .foregroundStyle(color)
        }
    }
}
