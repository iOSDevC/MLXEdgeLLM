// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MLXEdgeLLM",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
        .visionOS(.v1)
    ],
    products: [
        .library(name: "MLXEdgeLLM",      targets: ["MLXEdgeLLM"]),
        .library(name: "MLXEdgeLLMUI",    targets: ["MLXEdgeLLMUI"]),
        .library(name: "MLXEdgeLLMVoice", targets: ["MLXEdgeLLMVoice"]),
        .library(name: "MLXEdgeLLMDocs",  targets: ["MLXEdgeLLMDocs"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/ml-explore/mlx-swift-lm",
            branch: "main"
        )
    ],
    targets: [
        // MARK: - Core
        .target(
            name: "MLXEdgeLLM",
            dependencies: [
                .product(name: "MLXVLM",      package: "mlx-swift-lm"),
                .product(name: "MLXLLM",      package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ],
            path: "Sources/MLXEdgeLLM"
        ),
        
        // MARK: - UI
        .target(
            name: "MLXEdgeLLMUI",
            dependencies: [
                "MLXEdgeLLM",
                "MLXEdgeLLMVoice",   // enables VoiceTab inside ContentView
            ],
            path: "Sources/MLXEdgeLLMUI"
        ),
        
        // MARK: - Voice
        // Uses only Apple frameworks (Speech, AVFoundation) — no extra deps.
            .target(
                name: "MLXEdgeLLMVoice",
                dependencies: ["MLXEdgeLLM"],
                path: "Sources/MLXEdgeLLMVoice"
            ),
        
        // MARK: - Docs (RAG)
        // Uses only Apple frameworks (PDFKit) + network for embeddings API.
            .target(
                name: "MLXEdgeLLMDocs",
                dependencies: ["MLXEdgeLLM"],
                path: "Sources/MLXEdgeLLMDocs"
            ),
        
        // MARK: - Example App
        .target(
            name: "MLXEdgeLLMExample",
            dependencies: [
                "MLXEdgeLLM",
                "MLXEdgeLLMUI",
                "MLXEdgeLLMVoice",
                "MLXEdgeLLMDocs",
            ],
            path: "Sources/MLXEdgeLLMExample"
        ),
        
        // MARK: - Tests
        .testTarget(
            name: "MLXEdgeLLMTests",
            dependencies: ["MLXEdgeLLM"],
            path: "Tests/MLXEdgeLLMTests"
        )
    ]
)
