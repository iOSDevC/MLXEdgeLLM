// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "AuraLocal",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
        .visionOS(.v1)
    ],
    products: [
        .library(name: "AuraCore",              targets: ["AuraCore"]),
        .library(name: "AuraUI",                targets: ["AuraUI"]),
        .library(name: "AuraVoice",             targets: ["AuraVoice"]),
        .library(name: "AuraDocs",              targets: ["AuraDocs"]),
        .library(name: "AuraAppleIntelligence", targets: ["AuraAppleIntelligence"]),
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
            name: "AuraCore",
            dependencies: [
                .product(name: "MLXVLM",      package: "mlx-swift-lm"),
                .product(name: "MLXLLM",      package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ],
            path: "Sources/AuraCore"
        ),

        // MARK: - UI
        .target(
            name: "AuraUI",
            dependencies: [
                "AuraCore",
                "AuraVoice",
            ],
            path: "Sources/AuraUI"
        ),

        // MARK: - Voice
        .target(
            name: "AuraVoice",
            dependencies: ["AuraCore"],
            path: "Sources/AuraVoice"
        ),

        // MARK: - Docs (RAG)
        .target(
            name: "AuraDocs",
            dependencies: ["AuraCore"],
            path: "Sources/AuraDocs"
        ),

        // MARK: - Apple Intelligence Agents
        // Requires iOS 26+ / macOS 26+ with Apple Intelligence enabled.
        // No dependency on AuraCore — standalone module using FoundationModels.
            .target(
                name: "AuraAppleIntelligence",
                dependencies: [],
                path: "Sources/AuraAppleIntelligence",
                swiftSettings: [
                    .enableUpcomingFeature("StrictConcurrency")
                ]
            ),

        // MARK: - Example App
        .target(
            name: "AuraExample",
            dependencies: [
                "AuraCore",
                "AuraUI",
                "AuraVoice",
                "AuraDocs",
                "AuraAppleIntelligence",
            ],
            path: "Sources/AuraExample",
            exclude: ["Package.swift"]
        ),

        // MARK: - Tests
        .testTarget(
            name: "AuraCoreTests",
            dependencies: ["AuraCore"],
            path: "Tests/AuraCoreTests"
        )
    ]
)
