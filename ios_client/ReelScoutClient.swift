// ReelScout iOS Client - Share Extension + Main App
// Usage: Share Instagram Reel → ReelScout → Background processing → Local results

import Foundation
import UniformTypeIdentifiers

// ─── Models ──────────────────────────────────────────────────────────────
struct AnalyzeRequest: Codable {
    let vision: String
    let textLLM: String
    let resolve: String
    let caption: String
    
    enum CodingKeys: String, CodingKey {
        case vision
        case textLLM = "text_llm"
        case resolve
        case caption
    }
}

struct JobResponse: Codable {
    let jobID: String
    let status: String
    
    enum CodingKeys: String, CodingKey {
        case jobID = "job_id"
        case status
    }
}

struct JobStatus: Codable {
    let jobID: String
    let status: String
    let result: ReelResult?
    let error: String?
    let progress: String?
    
    enum CodingKeys: String, CodingKey {
        case jobID = "job_id"
        case status, result, error, progress
    }
}

struct ReelResult: Codable {
    let status: String
    let summary: String
    let candidates: [Candidate]
    let timings: [String: Double]
    let llm: LLMInfo
}

struct Candidate: Codable {
    let kind: String
    let name: String
    let url: String?
    let confidence: Double
    let verification: Verification
}

struct Verification: Codable {
    let ok: Bool
    let status: Int?
    let finalURL: String?
    let title: String?
    
    enum CodingKeys: String, CodingKey {
        case ok, status
        case finalURL = "final_url"
        case title
    }
}

struct LLMInfo: Codable {
    let provider: String
    let usage: Usage?
    let stages: [Stage]
}

struct Usage: Codable {
    let inputTokens: Int
    let outputTokens: Int
    let calls: Int
    
    enum CodingKeys: String, CodingKey {
        case inputTokens = "in"
        case outputTokens = "out"
        case calls
    }
}

struct Stage: Codable {
    let stage: String
    let model: String
    let inputTokens: Int?
    let outputTokens: Int?
    let calls: Int?
    
    enum CodingKeys: String, CodingKey {
        case stage, model
        case inputTokens = "in"
        case outputTokens = "out"
        case calls
    }
}

// ─── API Client ──────────────────────────────────────────────────────────
class ReelScoutAPI {
    let baseURL: URL
    let session: URLSession
    
    init(baseURL: String = "https://your-api.fly.dev") {
        self.baseURL = URL(string: baseURL)!
        self.session = URLSession(configuration: .default)
    }
    
    func analyzeVideo(
        videoURL: URL,
        request: AnalyzeRequest,
        progress: @escaping (Double) -> Void
    ) async throws -> JobResponse {
        // 1. Upload video
        let uploadURL = baseURL.appendingPathComponent("/api/v1/analyze")
        var request = URLRequest(url: uploadURL)
        request.httpMethod = "POST"
        
        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        var body = Data()
        func appendField(_ name: String, _ value: String) {
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n".data(using: .utf8)!)
            body.append("\(value)\r\n".data(using: .utf8)!)
        }
        
        appendField("vision", request.vision)
        appendField("text_llm", request.textLLM)
        appendField("resolve", request.resolve)
        appendField("caption", request.caption)
        
        // Video file
        let videoData = try Data(contentsOf: videoURL)
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"video\"; filename=\"\(videoURL.lastPathComponent)\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: video/mp4\r\n\r\n".data(using: .utf8)!)
        body.append(videoData)
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        
        request.httpBody = body
        
        let (data, response) = try await session.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            throw URLError(.badServerResponse)
        }
        
        return try JSONDecoder().decode(JobResponse.self, from: data)
    }
    
    func pollStatus(jobID: String) async throws -> JobStatus {
        let url = baseURL.appendingPathComponent("/api/v1/status/\(jobID)")
        let (data, _) = try await session.data(from: url)
        return try JSONDecoder().decode(JobStatus.self, from: data)
    }
    
    func waitForCompletion(jobID: String, interval: TimeInterval = 2.0) async throws -> ReelResult {
        while true {
            let status = try await pollStatus(jobID: jobID)
            switch status.status {
            case "finished":
                return status.result!
            case "failed":
                throw NSError(domain: "ReelScout", code: -1, userInfo: [NSLocalizedDescriptionKey: status.error ?? "Unknown error"])
            default:
                try await Task.sleep(nanoseconds: UInt64(interval * 1_000_000_000))
            }
        }
    }
}

// ─── Video Downloader (yt-dlp wrapper or native) ─────────────────────────
class VideoDownloader {
    static func downloadReel(url: String) async throws -> URL {
        // Option A: Use yt-dlp binary bundled in app (requires compilation for iOS)
        // Option B: Use native AVAssetExportSession for direct Instagram URLs (limited)
        // Option C: Call your server's /download endpoint that returns the video
        
        // Simplified: Call server to download and return video
        // In production, use yt-dlp compiled for iOS or a native Swift downloader
        throw NSError(domain: "VideoDownloader", code: -1, userInfo: [NSLocalizedDescriptionKey: "Implement yt-dlp or server-side download"])
    }
}

// ─── Local Storage ───────────────────────────────────────────────────────
class LocalResultStore {
    static let shared = LocalResultStore()
    private let queue = DispatchQueue(label: "com.reelscout.localstore")
    private let fileURL: URL
    
    private init() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        fileURL = docs.appendingPathComponent("reel_results.json")
    }
    
    func save(_ result: ReelResult, for jobID: String) {
        queue.async {
            var all = self.loadAll()
            all[jobID] = result
            try? JSONEncoder().encode(all).write(to: self.fileURL)
        }
    }
    
    func load(jobID: String) -> ReelResult? {
        queue.sync {
            return loadAll()[jobID]
        }
    }
    
    func loadAll() -> [String: ReelResult] {
        guard let data = try? Data(contentsOf: fileURL),
              let dict = try? JSONDecoder().decode([String: ReelResult].self, from: data) else {
            return [:]
        }
        return dict
    }
    
    func delete(jobID: String) {
        queue.async {
            var all = self.loadAll()
            all.removeValue(forKey: jobID)
            try? JSONEncoder().encode(all).write(to: self.fileURL)
        }
    }
}

// ─── Share Extension Handler ─────────────────────────────────────────────
/*
 // In ShareExtension/ShareViewController.swift
 import MobileCoreServices
 import UniformTypeIdentifiers
 
 class ShareViewController: UIViewController {
     override func viewDidLoad() {
         super.viewDidLoad()
         handleSharedContent()
     }
     
     func handleSharedContent() {
         guard let item = extensionContext?.inputItems.first as? NSExtensionItem,
               let provider = item.attachments?.first(where: { $0.hasItemConformingToTypeIdentifier(UTType.url.identifier) }) else {
             close()
             return
         }
         
         provider.loadItem(forTypeIdentifier: UTType.url.identifier, options: nil) { [weak self] url, _ in
             if let url = url as? URL, url.absoluteString.contains("instagram.com") {
                 Task {
                     await self?.processReel(url: url)
                 }
             } else {
                 self?.close()
             }
         }
     }
     
     func processReel(url: URL) async {
         let api = ReelScoutAPI()
         let request = AnalyzeRequest(vision: "auto", textLLM: "auto", resolve: "off", caption: "")
         
         do {
             // 1. Download video (or upload URL to server for download)
             let videoURL = try await VideoDownloader.downloadReel(url: url.absoluteString)
             
             // 2. Upload to server
             let job = try await api.analyzeVideo(videoURL: videoURL, request: request) { progress in
                 // Update UI with upload progress
             }
             
             // 3. Wait for result (or push notification)
             let result = try await api.waitForCompletion(jobID: job.jobID)
             
             // 4. Save locally
             LocalResultStore.shared.save(result, for: job.jobID)
             
             // 5. Cleanup local video
             try? FileManager.default.removeItem(at: videoURL)
             
             // 6. Notify user
             self?.showSuccess(result)
         } catch {
             self?.showError(error)
         }
     }
     
     func close() {
         extensionContext?.completeRequest(returningItems: [], completionHandler: nil)
     }
 }
 */

// ─── Usage in Main App ───────────────────────────────────────────────────
/*
 @main
 struct ReelScoutApp: App {
     var body: some Scene {
         WindowGroup {
             ContentView()
         }
     }
 }
 
 struct ContentView: View {
     @State private var results: [String: ReelResult] = [:]
     
     var body: some View {
         NavigationView {
             List {
                 ForEach(Array(results.keys.sorted()), id: \.self) { jobID in
                     if let result = results[jobID] {
                         NavigationLink(destination: ResultDetailView(result: result)) {
                             VStack(alignment: .leading) {
                                 Text(result.summary).lineLimit(2)
                                 Text(result.status).font(.caption).foregroundColor(.secondary)
                             }
                         }
                     }
                 }
             }
             .onAppear { loadResults() }
             .navigationTitle("ReelScout")
         }
     }
     
     func loadResults() {
         results = LocalResultStore.shared.loadAll()
     }
 }
 
 struct ResultDetailView: View {
     let result: ReelResult
     
     var body: some View {
         ScrollView {
             VStack(alignment: .leading, spacing: 16) {
                 Text(result.summary).font(.body)
                 
                 ForEach(result.candidates, id: \.name) { candidate in
                     VStack(alignment: .leading) {
                         HStack {
                             Text(candidate.kind.capitalized).font(.caption).padding(4).background(Color.blue.opacity(0.2)).cornerRadius(4)
                             Text(candidate.name).font(.headline)
                             Spacer()
                             Text("\(Int(candidate.confidence * 100))%").font(.caption).foregroundColor(.secondary)
                         }
                         if let url = candidate.url {
                             Link(url, destination: URL(string: url)!).font(.caption)
                         }
                         if candidate.verification.ok {
                             Label("Verified", systemImage: "checkmark.seal.fill").foregroundColor(.green).font(.caption)
                         }
                     }
                     .padding()
                     .background(Color(.systemGray6))
                     .cornerRadius(12)
                 }
                 
                 Text("Timings: \(result.timings)")
                     .font(.caption)
                     .foregroundColor(.secondary)
             }
             .padding()
         }
         .navigationTitle("Analysis")
     }
 }
 */