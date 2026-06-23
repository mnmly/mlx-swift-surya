import Foundation
import HuggingFace

/// Helpers for obtaining the `surya-ocr-2` checkpoint (`qwen3_5` VLM) from Hugging Face.
public enum SuryaModel {
    /// The bf16 reference repo id (`datalab-to/surya-ocr-2`, ~1.37 GB).
    public static let repoID = "datalab-to/surya-ocr-2"

    /// The HF hub cache folder name (`models--<owner>--<name>`).
    static var cacheFolderName: String {
        "models--" + repoID.replacingOccurrences(of: "/", with: "--")
    }

    /// Download the model to the default Hugging Face cache and return the local snapshot
    /// directory. Already-cached files are skipped, so this is also a cheap "ensure present"
    /// call. The snapshot's `preprocessor_config.json` is patched with `processor_class` so it
    /// loads without manual fixup (see ``patchProcessorClass(in:)``).
    ///
    /// - Parameters:
    ///   - force: re-download even when a complete copy is already cached.
    ///   - progressHandler: optional progress callback, delivered on the main actor.
    /// - Returns: the local snapshot directory (config.json + safetensors + tokenizer).
    @discardableResult
    public static func download(
        force: Bool = false,
        progressHandler: (@MainActor @Sendable (Progress) -> Void)? = nil
    ) async throws -> URL {
        if !force, let cached = cachedSnapshotDirectory() {
            if let progressHandler {
                let p = Progress(totalUnitCount: 1)
                p.completedUnitCount = 1
                await progressHandler(p)
            }
            try patchProcessorClass(in: cached)
            return cached
        }

        guard let repo = Repo.ID(rawValue: repoID) else {
            throw SuryaError.invalidRepoID(repoID)
        }
        let client = HubClient()
        let dir = try await client.downloadSnapshot(
            of: repo, revision: "main", progressHandler: progressHandler)
        try patchProcessorClass(in: dir)
        return dir
    }

    /// The local snapshot directory if a **complete** copy is already downloaded, else nil.
    /// Requires `config.json` + `tokenizer.json` + a `.safetensors` file so a half-finished
    /// download is not mistaken for a usable model.
    public static func cachedSnapshotDirectory() -> URL? {
        let base = URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent(".cache/huggingface/hub/\(cacheFolderName)/snapshots")
        let fm = FileManager.default
        guard let children = try? fm.contentsOfDirectory(at: base, includingPropertiesForKeys: nil)
        else { return nil }
        return children.first { dir in
            guard fm.fileExists(atPath: dir.appendingPathComponent("config.json").path),
                fm.fileExists(atPath: dir.appendingPathComponent("tokenizer.json").path)
            else { return false }
            let contents = (try? fm.contentsOfDirectory(atPath: dir.path)) ?? []
            return contents.contains { $0.hasSuffix(".safetensors") }
        }
    }

    /// Resolve a user-chosen directory to the actual model snapshot directory (the one that
    /// directly contains `config.json`). Handles pointing at a Hugging Face cache repo root
    /// (`…/models--datalab-to--surya-ocr-2`), whose files live under `snapshots/<commit>/`.
    public static func resolveSnapshotDirectory(_ url: URL) -> URL {
        let fm = FileManager.default
        if fm.fileExists(atPath: url.appendingPathComponent("config.json").path) { return url }

        let snapshots = url.appendingPathComponent("snapshots")
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: snapshots.path, isDirectory: &isDir), isDir.boolValue
        else { return url }

        func hasConfig(_ dir: URL) -> Bool {
            fm.fileExists(atPath: dir.appendingPathComponent("config.json").path)
        }

        if let ref = try? String(
            contentsOf: url.appendingPathComponent("refs/main"), encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines), !ref.isEmpty
        {
            let candidate = snapshots.appendingPathComponent(ref)
            if hasConfig(candidate) { return candidate }
        }

        let children = (try? fm.contentsOfDirectory(
            at: snapshots, includingPropertiesForKeys: [.contentModificationDateKey])) ?? []
        let valid = children.filter(hasConfig)
        let newest = valid.max { a, b in
            let da = (try? a.resourceValues(forKeys: [.contentModificationDateKey]))?
                .contentModificationDate ?? .distantPast
            let db = (try? b.resourceValues(forKeys: [.contentModificationDateKey]))?
                .contentModificationDate ?? .distantPast
            return da < db
        }
        return newest ?? url
    }

    /// The MLXVLM loader reads `processor_class` from `preprocessor_config.json`; surya-ocr-2
    /// declares it in `processor_config.json` instead. Copy it across (or default to the
    /// `qwen3_5` processor) so the snapshot loads without a *"Missing field 'processor_class'"*
    /// error. Idempotent.
    public static func patchProcessorClass(in directory: URL) throws {
        let url = directory.appendingPathComponent("preprocessor_config.json")
        guard let data = try? Data(contentsOf: url),
            var obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        else { return }
        if obj["processor_class"] == nil {
            obj["processor_class"] = processorClassFromConfig(in: directory) ?? "Qwen3VLProcessor"
            let out = try JSONSerialization.data(
                withJSONObject: obj, options: [.prettyPrinted, .sortedKeys])
            try out.write(to: url)
        }
    }

    private static func processorClassFromConfig(in directory: URL) -> String? {
        let url = directory.appendingPathComponent("processor_config.json")
        guard let data = try? Data(contentsOf: url),
            let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        else { return nil }
        return obj["processor_class"] as? String
    }
}
