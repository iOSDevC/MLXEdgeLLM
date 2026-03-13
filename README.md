<p align="center">
  <img src="Docs/Media/logo.png" alt="AuraLocal" width="200">
</p>

# AuraLocal

Lightweight on-device LLM & VLM Swift package for iOS/macOS, powered by MLX. Run Qwen3, Llama, Gemma, SmolVLM and other models locally — no API keys, no binary dependencies, fully private.

---

## v2.0 Highlights

- **Unified model management** — `ModelManager.shared.load()` provides LRU caching, in-flight deduplication, and automatic memory-pressure eviction across all tabs.
- **Swift 6 concurrency** — all public APIs are `@MainActor`-isolated or `Sendable`, with `actor`-based stores (`ConversationStore`, `DocumentLibrary`, `VectorStore`) for data-race safety.
- **OOM prevention** — adaptive memory budget based on `os_proc_available_memory()` (iOS) or physical RAM (macOS), with `DispatchSource` memory-pressure listeners that proactively evict LRU models before the OS kills the app.
- **Hybrid RAG pipeline** — FTS5 keyword pre-filter (top-20) followed by Accelerate-powered cosine re-ranking (top-5), all stored in a single SQLite database with BLOB vectors. Zero external dependencies.
- **Strict access control** — internal implementation details (`AuraEngine`, `ParsedDocument`, `DocumentExporter`, `LanguageDetector`) are hidden from the public API surface. Consumers use typed facades only.

---

## Requirements

- iOS 17+ / macOS 14+ / visionOS 1+
- Xcode 16+
- `Increased Memory Limit` entitlement (required for models > 500 MB)

---

## Installation

Add via Swift Package Manager:

```
https://github.com/iOSDevC/AuraLocal
```

Or in `Package.swift`:

```swift
.package(url: "https://github.com/iOSDevC/AuraLocal", branch: "main")
```

### Modules

| Module | Contents |
|--------|----------|
| `AuraCore` | Core inference, models, conversation persistence |
| `AuraUI` | SwiftUI views and ViewModels for drop-in UI |
| `AuraVoice` | Full-duplex voice interface (STT + TTS), 100% local |
| `AuraDocs` | RAG document library — PDF, DOCX, text, images |

```swift
// Core only
import AuraCore

// Core + prebuilt SwiftUI interface
import AuraCore
import AuraUI

// Core + voice (STT → LLM → TTS pipeline)
import AuraCore
import AuraVoice

// Core + document RAG
import AuraCore
import AuraDocs

// Everything
import AuraCore
import AuraUI
import AuraVoice
import AuraDocs
```

---

## Text Chat

```swift
import AuraCore

// One-liner
let reply = try await AuraLocal.chat("¿Cuánto gasté esta semana?")

// Reusable instance (loads model once — preferred for multiple calls)
let llm = try await AuraLocal.text(.qwen3_1_7b) { progress in
    print(progress) // "Downloading Qwen3 1.7B: 42%"
}
let reply = try await llm.chat("Summarize my expenses")

// Streaming
for try await token in llm.stream("Explain this transaction") {
    print(token, terminator: "")
}

// With system prompt
let reply = try await llm.chat(
    "What is the VAT rate in Mexico?",
    systemPrompt: "You are a personal finance assistant."
)
```

### Text Models

| Model | Size | Best for |
|-------|------|----------|
| `.qwen3_0_6b` | ~400 MB | Ultra-fast responses |
| `.qwen3_1_7b` ⭐ | ~1.0 GB | Balanced (default) |
| `.qwen3_4b` | ~2.5 GB | Higher quality |
| `.gemma3_1b` | ~700 MB | Google alternative |
| `.phi3_5_mini` | ~2.2 GB | Microsoft alternative |
| `.llama3_2_1b` | ~700 MB | Meta, lightweight |
| `.llama3_2_3b` | ~1.8 GB | Meta, higher quality |

---

## Vision / Image Analysis

```swift
import AuraCore

// One-liner receipt extraction
let json = try await AuraLocal.extractDocument(receiptImage)
// → {"store":"OXXO","date":"2026-03-06","items":[...],"total":125.50,"currency":"MXN"}

// Reusable instance
let vlm = try await AuraLocal.vision(.qwen35_0_8b) { print($0) }

// Free-form image analysis
let description = try await vlm.analyze("What items are on this receipt?", image: photo)

// Streaming with image
for try await token in vlm.streamVision("Describe this image", image: photo) {
    print(token, terminator: "")
}
```

### Vision Models

| Model | Size | Best for |
|-------|------|----------|
| `.qwen35_0_8b` ⭐ | ~625 MB | Default, iPhone |
| `.qwen35_2b` | ~1.7 GB | iPad, higher accuracy |
| `.smolvlm_500m` | ~1.0 GB | Minimum memory |
| `.smolvlm_2b` | ~1.5 GB | SmolVLM, balanced |

---

## OCR & Document Extraction

Specialized models optimized for receipts, invoices, and structured documents.

```swift
import AuraCore

// FastVLM — outputs structured JSON
let ocr = try await AuraLocal.specialized(.fastVLM_0_5b_fp16) { print($0) }
let json = try await ocr.extractDocument(receiptImage)

// Granite Docling — outputs DocTags, converted to Markdown
let docOCR = try await AuraLocal.specialized(.graniteDocling_258m)
let raw = try await docOCR.extractDocument(documentImage)
let markdown = AuraLocal.parseDocTags(raw)
```

### Specialized Models

| Model | Size | Output |
|-------|------|--------|
| `.fastVLM_0_5b_fp16` ⭐ | ~1.25 GB | JSON (receipts) |
| `.fastVLM_1_5b_int8` | ~800 MB | JSON (receipts) |
| `.graniteDocling_258m` | ~631 MB | DocTags → Markdown |
| `.graniteVision_3_3` | ~1.2 GB | Plain text |

---

## Receipt Scanner Example

```swift
import AuraCore

struct ReceiptData: Codable {
    let store: String
    let date: String
    let items: [Item]
    let subtotal: Double
    let tax: Double
    let total: Double
    let currency: String

    struct Item: Codable {
        let name: String
        let quantity: Int
        let price: Double
    }
}

func scanReceipt(_ image: PlatformImage) async throws -> ReceiptData {
    let json = try await AuraLocal.extractDocument(image)
    return try JSONDecoder().decode(ReceiptData.self, from: Data(json.utf8))
}
```

---

## Conversation Persistence

`ConversationStore` provides a SQLite-backed store (no external dependencies) for persisting chat history. The LLM automatically loads a context window of the most recent turns that fit within the token budget.

```swift
import AuraCore

let store = ConversationStore.shared

// Create a conversation
let conv = try await store.createConversation(model: .qwen3_1_7b, title: "Finance assistant")

// Chat with automatic history — context window managed automatically
let llm = try await AuraLocal.text(.qwen3_1_7b)
let reply  = try await llm.chat("What is 2+2?", in: conv.id)
let reply2 = try await llm.chat("Why?", in: conv.id) // includes previous exchange

// Streaming with history
for try await token in llm.stream("Tell me more", in: conv.id) {
    print(token, terminator: "")
}

// One-liner (creates conversation automatically)
let (reply, convID) = try await AuraLocal.chat("Hello", model: .qwen3_1_7b)

// List all conversations
let conversations = try await store.allConversations()

// Full-text search across all messages
let results = try await store.search("VAT Mexico")

// Auto-title based on first message
try await llm.autoTitle(conversationID: conv.id)

// Prune and summarize long conversations
try await llm.summarizeAndPrune(conversationID: conv.id)
```

### Context Window Management

When a conversation exceeds the token budget, `summarizeAndPrune` uses the model itself to summarize older turns and replace them with a compact system-level summary — preserving semantic continuity without truncating abruptly.

```swift
// Called automatically during chat if conversation exceeds 4096 tokens
try await llm.summarizeAndPrune(
    conversationID: conv.id,
    keepLastN: 10,         // always keep the 10 most recent turns
    maxContextTokens: 4096
)
```

---

## Voice Interface

`AuraVoice` provides a full-duplex voice pipeline using only Apple frameworks — no external dependencies, no network calls.

```
Microphone → SFSpeechRecognizer (on-device) → AuraLocal.stream() → AVSpeechSynthesizer
```

Sentences are streamed to TTS **while the LLM is still generating** — the assistant starts speaking after the first complete sentence, not after the full response.

Language is detected automatically per utterance using `NLLanguageRecognizer` and mapped to the best available system voice with region (e.g. `"es"` → `"es-MX"`).

### Drop-in button

```swift
import AuraVoice

// Minimal — manages its own VoiceSession internally
VoiceButton(llm: llm)

// With external session for full state control
@StateObject var session = VoiceSession(llm: llm)

VoiceButton(session: session)
Text(session.transcript)  // live STT transcript
Text(session.response)    // live LLM response
```

### Full voice chat view

```swift
import AuraVoice

// Complete UI: transcript bubble + response bubble + VoiceButton
VoiceChatView(llm: llm)

// With persistent conversation
VoiceChatView(llm: llm, conversationID: conv.id)
```

### Manual pipeline control

```swift
import AuraVoice

let session = VoiceSession(llm: llm, conversationID: conv.id)

// Request permissions once on launch
let granted = await session.requestPermissions()

// Start — silence detection triggers LLM automatically
try await session.startListening()

// Or stop manually
await session.stopListening()

// Interrupt TTS mid-sentence
session.interrupt()

// Cancel everything
session.cancel()
```

### Configuration

```swift
var config = VoiceSession.Config()
config.silenceThreshold     = 1.4    // seconds of silence before triggering LLM
config.maxRecordingDuration = 30     // max recording time in seconds
config.speakingRate         = 0.5    // TTS rate (0–1)
config.maxTokens            = 512    // max LLM tokens per response
config.systemPrompt         = "You are a helpful assistant. Be concise."

let session = VoiceSession(llm: llm, config: config)
```

### VoiceSession States

| State | Meaning |
|-------|---------|
| `.idle` | Ready, waiting for input |
| `.listening` | Recording + live transcription |
| `.thinking(partial:)` | LLM streaming, partial response available |
| `.speaking(sentence:)` | TTS playing current sentence |
| `.error(String)` | Something went wrong |

### Required permissions

Add to your `Info.plist`:

```xml
<key>NSSpeechRecognitionUsageDescription</key>
<string>Used for voice input to the local AI assistant.</string>
<key>NSMicrophoneUsageDescription</key>
<string>Used to capture your voice for the AI assistant.</string>
```

---

## Document Library (RAG)

`AuraDocs` provides a fully local Retrieval-Augmented Generation (RAG) pipeline. Index documents once, then ask questions in natural language. No API keys, no cloud services.

### Supported formats

| Format | Parser |
|--------|--------|
| `.pdf` | PDFKit (text extraction per page) |
| `.docx` | ZIP + XML (no external dependencies) |
| `.txt`, `.md`, `.markdown` | Plain text |
| `.png`, `.jpg`, `.jpeg`, `.heic`, `.tiff` | MLX VLM OCR |

### Retrieval pipeline

```
query → TF-IDF embed → FTS5 top-20 candidates → cosine re-rank top-5 → LLM
```

Two-stage hybrid search: FTS5 for fast keyword recall, cosine similarity for semantic precision. All vectors stored as BLOBs in SQLite — no external vector database required.

### Embedding backend

`AutoEmbeddingProvider` uses TF-IDF sparse embeddings with IDF weights built from the indexed corpus — 100% local, zero downloads, works on device and simulator. Designed to upgrade transparently to dense MLX embeddings when `mlx-swift-lm` exposes that API.

### Quick start

```swift
import AuraCore
import AuraDocs

// 1. Configure once (e.g. in app startup)
let llm      = try await AuraLocal.text(.qwen3_1_7b)
let embedder = AutoEmbeddingProvider()

let library = DocumentLibrary.shared
await library.configure(embeddingProvider: embedder, llm: llm)
try await library.open()

// 2. Index documents — progress delivered on @MainActor
try await library.add(url: pdfURL) { progress in
    print(progress) // "Embedding MyDoc: 42%"
}
try await library.add(url: docxURL)
try await library.add(url: imageURL)   // OCR via VLM

// Rebuild TF-IDF weights after indexing
await library.refreshCorpus()

// 3. Ask questions
let answer = try await library.ask("What is the contract amount?")
print(answer.text)

// 4. Inspect sources
for source in answer.sources {
    print("[\(source.documentTitle) p.\(source.pageNumber)] score: \(source.score)")
    print(source.excerpt)
}
```

### Stateful document chat

```swift
import AuraDocs

// DocumentChat maintains conversation history and cites sources per message
let chat = DocumentChat(library: library, llm: llm)

let reply1 = try await chat.send("What is the payment schedule?")
let reply2 = try await chat.send("And the penalties for late payment?") // context-aware

for msg in chat.messages {
    print(msg.role, msg.text)
    print(msg.sources.map { $0.documentTitle }) // cited documents
}
```

### Advanced options

```swift
// Custom chunk size and overlap
let library = DocumentLibrary(
    chunkTargetTokens:    512,   // target tokens per chunk
    chunkOverlapFraction: 0.1    // 10% overlap between chunks
)

// Ask with more context
let answer = try await library.ask(
    "Summarize the key obligations",
    topK:             8,      // retrieve 8 chunks (default 5)
    maxContextTokens: 4096,   // context budget for LLM
    systemPrompt:     "You are a legal assistant. Be precise and cite page numbers."
)

// Manage library
let docs = try await library.allDocuments()
try await library.removeDocument(id: doc.id)
```

### Progress stages

`onProgress` is delivered on the `@MainActor` and reports four stages:

| Stage | Example message | Approx. % |
|-------|----------------|-----------|
| Parsing | `"Parsing MyDoc.pdf…"` | 5% |
| Chunking | `"Chunking MyDoc…"` | 15% |
| Embedding | `"Embedding MyDoc: 42%"` | 15–100% |
| Done | `"'MyDoc' indexed ✓ (253 chunks)"` | 100% |

### Drop-in tab

Add `DocsTab` to any existing `TabView`:

```swift
import AuraDocs

TabView {
    // ... existing tabs
    DocsTab()
        .tabItem { Label("Docs", systemImage: "doc.text.magnifyingglass") }
}
```

`DocsTab` includes a file picker (multi-select), per-document progress bar with percentage, swipe-to-delete, and a full chat sheet with expandable source citations.

---

## Prebuilt SwiftUI Interface

`AuraUI` provides a ready-to-use tabbed interface. Add `AuraVoice` to unlock the Voice tab.

```swift
import SwiftUI
import AuraUI
import AuraVoice  // enables Voice tab

@main
struct MyApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
```

| Tab | Module | Description |
|-----|--------|-------------|
| **Text** | `AuraUI` | Persistent multi-conversation chat with streaming |
| **Vision** | `AuraUI` | Image analysis with standard and streaming modes |
| **OCR** | `AuraUI` | Document and receipt extraction |
| **Models** | `AuraUI` | Browser showing all models and download status |
| **Voice** | `AuraVoice` | Full-duplex voice chat with auto language detection |
| **Docs** | `AuraDocs` | Document library and RAG chat |

---

## Model Discovery

```swift
import AuraCore

// Filtered collections — downloaded models sorted first
let textModels        = Model.textModels
let visionModels      = Model.visionModels
let specializedModels = Model.specializedModels

// Check download status
if Model.qwen3_1_7b.isDownloaded {
    print("Ready at: \(Model.qwen3_1_7b.cacheDirectory.path)")
}

// Model metadata
let model = Model.qwen3_1_7b
print(model.displayName)       // "Qwen3 1.7B"
print(model.approximateSizeMB) // 1000
print(model.purpose)           // .text
```

---

## Model Management

`ModelManager` is the recommended way to load models. It prevents redundant downloads, shares instances across tabs, and handles memory pressure automatically.

```swift
import AuraCore

// Load from anywhere — returns cached instance if already loaded
let llm = try await ModelManager.shared.load(.qwen3_1_7b)

// Observe per-model state in SwiftUI
@ObservedObject var manager = ModelManager.shared

switch manager.state(for: .qwen3_1_7b) {
case .idle:                     // not loaded
case .downloading(let progress): // "42% — Qwen3 1.7B"
case .loading:                  // downloaded, loading into RAM
case .ready:                    // ready for inference
case .failed(let error):        // load failed
}

// Manual eviction
ModelManager.shared.evict(.qwen3_1_7b)
ModelManager.shared.evictAll()
```

### Memory Budget

The LRU cache size adapts to the device:

| Device RAM | Budget | Behavior |
|-----------|--------|----------|
| < 4 GB | 1 model | Evicts on every model switch |
| 4–6 GB | 1–2 models | iPhone 15, base iPad |
| 8+ GB | 2–4 models | iPad Pro, Mac |

When the OS sends a memory warning (`DispatchSource.makeMemoryPressureSource` + `UIApplication.didReceiveMemoryWarningNotification`), all models except the most recently used are evicted immediately.

---

## Entitlements

Add to your `.entitlements` file for models larger than 500 MB:

```xml
<key>com.apple.developer.kernel.increased-memory-limit</key>
<true/>
```

---

## Concurrency Model

AuraLocal is designed for Swift 6 strict concurrency:

| Type | Isolation | Rationale |
|------|-----------|-----------|
| `AuraLocal` | `@MainActor` | Wraps MLX callbacks that must fire on main thread |
| `ModelManager` | `@MainActor` | `ObservableObject` publishing `@Published` state |
| `ConversationStore` | `actor` | Serializes SQLite reads/writes without locks |
| `DocumentLibrary` | `actor` | Coordinates parsing, embedding, and vector store |
| `VectorStore` | `actor` | Owns the SQLite connection for vector operations |
| `VoiceSession` | `@MainActor` | Drives `AVAudioEngine` + `SFSpeechRecognizer` on main |
| `TFIDFEmbeddingProvider` | `actor` | Mutable IDF state updated from concurrent indexing |
| `Model`, `Turn`, `Conversation` | `Sendable` | Value types safe to pass across isolation boundaries |

All streaming APIs use `AsyncThrowingStream` to bridge MLX's callback-based inference to Swift async/await.

---

## Architecture

```
AuraCore
├── AuraLocal.text()        →  AuraEngine  →  MLXLLM
├── AuraLocal.vision()      →  AuraEngine  →  MLXVLM
├── AuraLocal.specialized() →  AuraEngine  →  MLXVLM
├── ConversationStore      →  SQLite (no external deps)
└── AuraLocal+History       →  context window · auto-title · pruning

AuraUI (optional)
├── ContentView  (TabView)
├── TextChatTab  →  TextChatViewModel  →  ConversationStore
├── VisionTab    →  VisionViewModel
├── OCRTab       →  OCRViewModel
└── ModelsTab

AuraVoice (optional)
├── VoiceSession             →  SFSpeechRecognizer (on-device STT)
│                            →  AuraLocal.stream() + ConversationStore
│                            →  AVSpeechSynthesizer (on-device TTS)
├── VoiceButton              →  SwiftUI mic button with state animations
├── VoiceChatView            →  Full voice chat UI
└── VoiceTab                 →  Tab for AuraUI ContentView

AuraDocs (optional)
├── DocumentLibrary          →  add() · ask() · allDocuments() · refreshCorpus()
├── DocumentParserDispatcher →  PDF (PDFKit) · DOCX (ZIP+XML) · TXT · Image (VLM OCR)
├── DocumentChunker          →  sliding window · sentence boundaries · overlap
├── AutoEmbeddingProvider    →  TF-IDF sparse (local, no download)
│   └── TFIDFEmbeddingProvider  →  DJB2 hash buckets · IDF weights · cosine
├── VectorStore              →  SQLite BLOB vectors · FTS5 pre-filter · cosine re-rank
├── DocumentChat             →  stateful Q&A · source citations · ConversationStore
└── DocsTab                  →  SwiftUI tab · file picker · progress bar · chat sheet

Sources/
├── AuraCore/
├── AuraUI/
├── AuraVoice/
├── AuraDocs/
└── AuraExample/

All models download automatically on first use and are cached at:
  ~/Library/Caches/models/<org>/<repo>/
```

---

## License

MIT
