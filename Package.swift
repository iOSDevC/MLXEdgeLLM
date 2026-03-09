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
        .library(name: "MLXEdgeLLM",                   targets: ["MLXEdgeLLM"]),
        .library(name: "MLXEdgeLLMUI",                 targets: ["MLXEdgeLLMUI"]),
        .library(name: "MLXEdgeLLMVoice",              targets: ["MLXEdgeLLMVoice"]),
        .library(name: "MLXEdgeLLMDocs",               targets: ["MLXEdgeLLMDocs"]),
        .library(name: "MLXEdgeLLMAppleIntelligence",  targets: ["MLXEdgeLLMAppleIntelligence"]),
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
                "MLXEdgeLLMVoice",
            ],
            path: "Sources/MLXEdgeLLMUI"
        ),
        
        // MARK: - Voice
        .target(
            name: "MLXEdgeLLMVoice",
            dependencies: ["MLXEdgeLLM"],
            path: "Sources/MLXEdgeLLMVoice"
        ),
        
        // MARK: - Docs (RAG)
        .target(
            name: "MLXEdgeLLMDocs",
            dependencies: ["MLXEdgeLLM"],
            path: "Sources/MLXEdgeLLMDocs"
        ),
        
        // MARK: - Apple Intelligence Agents
        // Requires iOS 26+ / macOS 26+ with Apple Intelligence enabled.
        // No dependency on MLXEdgeLLM — standalone module using FoundationModels.
            .target(
                name: "MLXEdgeLLMAppleIntelligence",
                dependencies: [],
                path: "Sources/MLXEdgeLLMAppleIntelligence",
                swiftSettings: [
                    .enableUpcomingFeature("StrictConcurrency")
                ]
            ),
        
        // MARK: - Example App
        .target(
            name: "MLXEdgeLLMExample",
            dependencies: [
                "MLXEdgeLLM",
                "MLXEdgeLLMUI",
                "MLXEdgeLLMVoice",
                "MLXEdgeLLMDocs",
                "MLXEdgeLLMAppleIntelligence",
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
