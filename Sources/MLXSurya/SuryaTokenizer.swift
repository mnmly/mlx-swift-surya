import Foundation
import MLXLMCommon

// Import only the two Jinja types we need — a bare `import Jinja` also brings Jinja's own
// `Tokenizer` lexer type into scope, colliding with `MLXLMCommon.Tokenizer`.
import struct Jinja.Template
import enum Jinja.Value

/// A char-level `WordLevel` tokenizer for surya-ocr-2, conforming to ``MLXLMCommon/Tokenizer``.
///
/// surya-ocr-2 ships a Hugging Face `WordLevel` tokenizer whose `pre_tokenizer` splits text
/// into **individual characters** (`Split(Regex ".")`) and whose `decoder` is `Fuse`
/// (concatenation). swift-transformers has no `WordLevel` implementation — its registry falls
/// back to `BPETokenizer`, which fatal-errors on the missing `merges` table. This type
/// implements MLXVLM's tokenizer protocol directly and is injected via ``SuryaTokenizerLoader``.
///
/// Encoding peels the 1948 special/added tokens (`<|im_start|>`, `<|image_pad|>`, …) as whole
/// units (longest-first), then maps every remaining Unicode scalar to its vocab id (or `unk`).
/// Decoding fuses id→token strings. Chat templating renders `chat_template.jinja` via Jinja,
/// mirroring `PreTrainedTokenizer.applyChatTemplate`.
public final class SuryaWordLevelTokenizer: @unchecked Sendable, Tokenizer {
    private let vocab: [String: Int]
    private let idToTokenMap: [Int: String]
    private let specialIds: Set<Int>
    private let addedTokensRegex: NSRegularExpression?
    private let chatTemplate: String?

    public let bosToken: String?
    public let eosToken: String?
    public let unknownToken: String?

    /// Build the tokenizer from a model snapshot directory containing `tokenizer.json`,
    /// `tokenizer_config.json`, and (optionally) `chat_template.jinja`.
    public init(directory: URL) throws {
        let tokData = try JSONSerialization.jsonObject(
            with: Data(contentsOf: directory.appendingPathComponent("tokenizer.json")))
        guard let tok = tokData as? [String: Any],
            let model = tok["model"] as? [String: Any],
            let rawVocab = model["vocab"] as? [String: Any]
        else { throw SuryaError.missingModelFile("tokenizer.json (model.vocab)") }

        var vocab: [String: Int] = [:]
        vocab.reserveCapacity(rawVocab.count)
        for (k, v) in rawVocab {
            if let i = (v as? NSNumber)?.intValue { vocab[k] = i }
        }
        self.vocab = vocab
        var rev: [Int: String] = [:]
        rev.reserveCapacity(vocab.count)
        for (k, i) in vocab { rev[i] = k }
        self.idToTokenMap = rev

        // Added tokens: collect contents (for the peel regex) and special-token ids.
        var specialIds: Set<Int> = []
        var addedContents: [String] = []
        if let added = tok["added_tokens"] as? [[String: Any]] {
            for entry in added {
                guard let content = entry["content"] as? String else { continue }
                addedContents.append(content)
                if (entry["special"] as? Bool) == true, let id = (entry["id"] as? NSNumber)?.intValue {
                    specialIds.insert(id)
                }
            }
        }
        self.specialIds = specialIds

        // Longest-first alternation so `<|image_pad|>` wins over any shorter prefix.
        addedContents.sort { $0.count > $1.count }
        if addedContents.isEmpty {
            self.addedTokensRegex = nil
        } else {
            let pattern = addedContents.map { NSRegularExpression.escapedPattern(for: $0) }
                .joined(separator: "|")
            self.addedTokensRegex = try? NSRegularExpression(pattern: pattern)
        }

        // Special-token strings from tokenizer_config.json.
        let cfgData =
            (try? JSONSerialization.jsonObject(
                with: Data(contentsOf: directory.appendingPathComponent("tokenizer_config.json"))))
            as? [String: Any] ?? [:]
        func tokenString(_ key: String) -> String? {
            if let s = cfgData[key] as? String { return s }
            if let d = cfgData[key] as? [String: Any], let c = d["content"] as? String { return c }
            return nil
        }
        self.bosToken = tokenString("bos_token")
        self.eosToken = tokenString("eos_token")
        self.unknownToken = tokenString("unk_token")

        // Chat template: surya keeps it in chat_template.jinja (not tokenizer_config.json).
        let jinjaURL = directory.appendingPathComponent("chat_template.jinja")
        if let s = try? String(contentsOf: jinjaURL, encoding: .utf8) {
            self.chatTemplate = s
        } else {
            self.chatTemplate = cfgData["chat_template"] as? String
        }
    }

    private var unknownId: Int { unknownToken.flatMap { vocab[$0] } ?? 0 }

    // MARK: - MLXLMCommon.Tokenizer

    /// Split into tokens: peel special/added tokens whole, char-split the rest.
    func tokenize(_ text: String) -> [String] {
        var tokens: [String] = []
        let ns = text as NSString
        var last = 0
        if let re = addedTokensRegex {
            re.enumerateMatches(in: text, range: NSRange(location: 0, length: ns.length)) { m, _, _ in
                guard let m else { return }
                let r = m.range
                if r.location > last {
                    let gap = ns.substring(with: NSRange(location: last, length: r.location - last))
                    tokens.append(contentsOf: gap.unicodeScalars.map { String($0) })
                }
                tokens.append(ns.substring(with: r))
                last = r.location + r.length
            }
        }
        if last < ns.length {
            tokens.append(contentsOf: ns.substring(from: last).unicodeScalars.map { String($0) })
        }
        return tokens
    }

    public func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        // surya's post_processor adds no special tokens, so `addSpecialTokens` is a no-op.
        let unk = unknownId
        return tokenize(text).map { vocab[$0] ?? unk }
    }

    public func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        var out = ""
        for id in tokenIds {
            if skipSpecialTokens && specialIds.contains(id) { continue }
            if let t = idToTokenMap[id] { out += t }
        }
        return out
    }

    public func convertTokenToId(_ token: String) -> Int? { vocab[token] }
    public func convertIdToToken(_ id: Int) -> String? { idToTokenMap[id] }

    public func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        guard let templateString = chatTemplate else { throw TokenizerError.missingChatTemplate }
        let template = try Template(templateString, with: .init(lstripBlocks: true, trimBlocks: true))
        var context: [String: Value] = [
            "messages": .array(try messages.map { try Value(any: $0) }),
            "add_generation_prompt": .boolean(true),
        ]
        if let tools { context["tools"] = .array(try tools.map { try Value(any: $0) }) }
        if let additionalContext {
            for (k, v) in additionalContext { context[k] = try Value(any: v) }
        }
        if let bosToken { context["bos_token"] = .string(bosToken) }
        if let eosToken { context["eos_token"] = .string(eosToken) }
        if let unknownToken { context["unk_token"] = .string(unknownToken) }

        let rendered = try template.render(context)
        return encode(text: rendered, addSpecialTokens: false)
    }
}

/// A ``MLXLMCommon/TokenizerLoader`` that builds a ``SuryaWordLevelTokenizer`` — injected into
/// `VLMModelFactory.loadContainer(from:using:)` so surya-ocr-2's char-level WordLevel tokenizer
/// loads instead of the default loader (which would fall back to BPE and crash).
public struct SuryaTokenizerLoader: TokenizerLoader {
    public init() {}
    public func load(from directory: URL) async throws -> any Tokenizer {
        try SuryaWordLevelTokenizer(directory: directory)
    }
}
