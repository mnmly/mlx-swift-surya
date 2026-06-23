import Foundation
import NaturalLanguage

// Post-processing that turns the VLM's per-block HTML (``OCRResult``) into a clean,
// reading-ordered document of headings / paragraphs / lists / tables. This is pure
// Swift — it touches no MLX op — so it runs anywhere (including `swift test`) and is
// shared by the CLI and a SwiftUI frontend via ``SuryaSession/structure(pages:options:)``.
//
// Why this stage exists: surya-ocr-2 already returns reading-ordered, label-tagged
// blocks and joins wrapped lines *inside* each `<p>`, so intra-paragraph chopping is
// gone. What remains is everything *between* blocks — a paragraph split across a page
// or column break, end-of-block hyphenation, and running headers/footers interleaved
// in reading order. ``Structurer`` stitches those back together (script-aware, so it
// works beyond English), then segments whole paragraphs into sentences.

/// One element of a ``StructuredDocument``: a heading, a paragraph, a list, a table, an
/// equation, or a figure placeholder. Encoded as flat JSON with a `type` discriminator.
public enum DocElement: Sendable, Equatable {
    /// A heading at `level` (1–6) with its plain text.
    case heading(level: Int, text: String)
    /// A body paragraph (stitched whole and segmented into ``Paragraph/sentences``).
    case paragraph(Paragraph)
    /// A list; `ordered` distinguishes `<ol>` from `<ul>`.
    case list(ordered: Bool, items: [String])
    /// A table, kept as its source HTML (structure preserved, not flattened to prose).
    case table(html: String)
    /// An equation, kept as its source HTML/MathML.
    case equation(html: String)
    /// A figure/picture region with an optional caption.
    case figure(caption: String?)
}

/// A body paragraph after stitching, with both a faithful transcription and (optionally) a
/// normalized form, its detected language, and its sentence segmentation.
public struct Paragraph: Sendable, Codable, Equatable {
    /// The faithful transcription exactly as recognized (long-s, ligatures, etc. preserved).
    public var text: String
    /// A normalized rendering (long-s→s, typographic ligatures folded, NFC) when
    /// ``Structurer/Options/normalizeOrthography`` is enabled; otherwise `nil`.
    public var normalizedText: String?
    /// The paragraph split into sentences (the whole, un-chopped paragraph as input).
    public var sentences: [String]
    /// Detected dominant language as a BCP-47 code (e.g. `"it"`, `"en"`), or `nil` when
    /// the text is too short to detect reliably.
    public var language: String?
    /// The canonical source-block label this paragraph came from (e.g. `"Text"`, `"Caption"`).
    public var label: String

    public init(
        text: String, normalizedText: String? = nil, sentences: [String] = [],
        language: String? = nil, label: String = "Text"
    ) {
        self.text = text
        self.normalizedText = normalizedText
        self.sentences = sentences
        self.language = language
        self.label = label
    }
}

/// A reading-ordered, structured document produced by ``Structurer``.
public struct StructuredDocument: Sendable, Codable, Equatable {
    /// The document's elements in reading order.
    public var elements: [DocElement]

    public init(elements: [DocElement]) {
        self.elements = elements
    }

    /// Render the document as Markdown (headings as `#`, lists as `-`/`1.`, tables/equations
    /// passed through as raw HTML, figures as `*[figure]*`).
    public func markdown() -> String {
        elements.map { el -> String in
            switch el {
            case .heading(let level, let text):
                return String(repeating: "#", count: min(max(level, 1), 6)) + " " + text
            case .paragraph(let p):
                return p.text
            case .list(let ordered, let items):
                return items.enumerated().map { idx, item in
                    ordered ? "\(idx + 1). \(item)" : "- \(item)"
                }.joined(separator: "\n")
            case .table(let html), .equation(let html):
                return html
            case .figure(let caption):
                return caption.map { "*[figure: \($0)]*" } ?? "*[figure]*"
            }
        }.joined(separator: "\n\n")
    }
}

/// Turns one or more ``OCRResult`` pages into a ``StructuredDocument``: routes blocks by
/// label, stitches prose split across block/page boundaries (de-hyphenating word wraps),
/// drops running headers/footers, detects each paragraph's language, and segments it into
/// sentences. Model-free and `Sendable` — construct once and reuse.
///
/// The stitcher is script-aware: it relies on sentence-terminal punctuation (Latin `.!?`
/// plus CJK/Arabic/Devanagari terminators) rather than capitalization, so it works for
/// non-English text. Title pages and display type stay as headings/figures because routing
/// keys off the block ``OCRBlock/label`` — sentence logic never runs on them.
public struct Structurer: Sendable {
    /// Knobs for ``Structurer``. Defaults suit running prose; enable
    /// ``normalizeOrthography`` for early-modern/historical print.
    public struct Options: Sendable {
        /// Produce a normalized ``Paragraph/normalizedText`` (long-s→s, ligature folding,
        /// NFC) alongside the faithful ``Paragraph/text``. Default `false`.
        public var normalizeOrthography: Bool
        /// Join words split by an end-of-block wrap hyphen, dropping the hyphen. Default `true`.
        public var dehyphenate: Bool
        /// Segment each stitched paragraph into ``Paragraph/sentences`` via `NLTokenizer`.
        /// Default `true`.
        public var segmentSentences: Bool
        /// Drop `PageHeader`/`PageFooter` blocks (they otherwise break a cross-page stitch).
        /// Default `true`.
        public var dropRunningHeaders: Bool

        public init(
            normalizeOrthography: Bool = false, dehyphenate: Bool = true,
            segmentSentences: Bool = true, dropRunningHeaders: Bool = true
        ) {
            self.normalizeOrthography = normalizeOrthography
            self.dehyphenate = dehyphenate
            self.segmentSentences = segmentSentences
            self.dropRunningHeaders = dropRunningHeaders
        }
    }

    /// The options this structurer applies.
    public let options: Options

    public init(options: Options = .init()) {
        self.options = options
    }

    /// Structure a single OCR'd page.
    public func structure(_ page: OCRResult) -> StructuredDocument {
        structure([page])
    }

    /// Structure a sequence of OCR'd pages (in page order), stitching paragraphs that run
    /// across the page boundary.
    public func structure(_ pages: [OCRResult]) -> StructuredDocument {
        // 1. Flatten blocks across pages, each page sorted by reading order.
        var candidates: [Candidate] = []
        for page in pages {
            let ordered = page.blocks.sorted { $0.readingOrder < $1.readingOrder }
            for block in ordered { candidates.append(contentsOf: self.candidates(for: block)) }
        }

        // 2. Walk candidates, stitching consecutive continuable prose into open paragraphs.
        var elements: [DocElement] = []
        var open: ProseAccumulator?

        func flush() {
            guard let acc = open else { return }
            elements.append(.paragraph(finalizeParagraph(text: acc.text, label: acc.label)))
            open = nil
        }

        for candidate in candidates {
            switch candidate {
            case .prose(let text, let label, let continuable, let preformatted):
                if preformatted {
                    flush()
                    elements.append(
                        .paragraph(finalizeParagraph(text: text, label: label, segment: false)))
                } else if var acc = open, acc.continuable, continuable,
                    let joined = joinIfContinuation(acc.text, text)
                {
                    acc.text = joined
                    open = acc
                } else {
                    flush()
                    open = ProseAccumulator(text: text, label: label, continuable: continuable)
                }
            case .heading(let level, let text):
                flush()
                elements.append(.heading(level: level, text: text))
            case .list(let ordered, let items):
                flush()
                elements.append(.list(ordered: ordered, items: items))
            case .table(let html):
                flush()
                elements.append(.table(html: html))
            case .equation(let html):
                flush()
                elements.append(.equation(html: html))
            case .figure(let caption):
                flush()
                elements.append(.figure(caption: caption))
            }
        }
        flush()
        return StructuredDocument(elements: elements)
    }

    // MARK: - Routing

    /// Convert one OCR block into zero or more structuring candidates based on its
    /// canonical ``OCRBlock/label``.
    private func candidates(for block: OCRBlock) -> [Candidate] {
        let label = block.label
        let html = block.html

        // Skipped blocks (Picture/Figure/Diagram) and empties carry no text.
        if block.skipped || html.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            switch label {
            case "Picture", "Figure", "Diagram", "ChemicalBlock", "Form":
                return [.figure(caption: nil)]
            default:
                return []
            }
        }

        switch label {
        case "PageHeader", "PageFooter":
            if options.dropRunningHeaders { return [] }
            let text = HTMLText.plainText(html)
            return text.isEmpty
                ? [] : [.prose(text: text, label: label, continuable: false, preformatted: false)]

        case "SectionHeader", "TableOfContents":
            let text = HTMLText.plainText(html)
            return text.isEmpty ? [] : [.heading(level: HTMLText.headingLevel(html) ?? 2, text: text)]

        case "Table", "Form":
            return [.table(html: SuryaParsers.cleanBlockHTML(html))]

        case "Equation":
            return [.equation(html: SuryaParsers.cleanBlockHTML(html))]

        case "ListGroup":
            let items = HTMLText.listItems(html)
            if items.isEmpty {
                let text = HTMLText.plainText(html)
                return text.isEmpty
                    ? []
                    : [.prose(text: text, label: label, continuable: false, preformatted: false)]
            }
            return [.list(ordered: html.range(of: "<ol", options: .caseInsensitive) != nil, items: items)]

        case "Picture", "Figure", "Diagram", "ChemicalBlock":
            return [.figure(caption: nil)]

        case "Code":
            let text = HTMLText.plainText(html, preserveBreaks: true)
            return text.isEmpty
                ? [] : [.prose(text: text, label: label, continuable: false, preformatted: true)]

        default:
            // Text / Caption / Footnote / Bibliography and any unknown text-bearing label.
            let continuable = ["Text", "Footnote", "Bibliography"].contains(label)
            return HTMLText.paragraphs(html).map {
                .prose(text: $0, label: label, continuable: continuable, preformatted: false)
            }
        }
    }

    // MARK: - Stitching

    /// If `cur` continues the sentence in `prev`, return the joined text; otherwise `nil`
    /// (a paragraph boundary). A trailing wrap-hyphen joins (de-hyphenating when enabled);
    /// otherwise text that does not end in sentence-terminal punctuation is treated as open.
    private func joinIfContinuation(_ prev: String, _ cur: String) -> String? {
        let p = prev.trimmedTrailingWhitespace()
        let c = cur.trimmedLeadingWhitespace()
        if c.isEmpty { return p }

        if let last = p.last, last == "-" || last == "\u{00AD}" {
            let beforeIdx = p.index(p.endIndex, offsetBy: -2, limitedBy: p.startIndex)
            let letterBefore = beforeIdx.map { p[$0].isLetter } ?? false
            if letterBefore {
                return options.dehyphenate ? String(p.dropLast()) + c : p + c
            }
        }
        if endsWithSentenceTerminator(p) { return nil }
        return p + " " + c
    }

    // MARK: - Finalizing a paragraph

    private func finalizeParagraph(text: String, label: String, segment: Bool = true) -> Paragraph {
        let normalized = options.normalizeOrthography ? normalizeOrthography(text) : nil
        let basis = normalized ?? text
        let language = detectLanguage(basis)
        let sentences: [String]
        if segment && options.segmentSentences {
            sentences = sentenceSplit(text, language: language)
        } else {
            sentences = [text]
        }
        return Paragraph(
            text: text, normalizedText: normalized, sentences: sentences,
            language: language?.rawValue, label: label)
    }

    private func detectLanguage(_ text: String) -> NLLanguage? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.count >= 12 else { return nil }  // too short to detect reliably
        let recognizer = NLLanguageRecognizer()
        recognizer.processString(trimmed)
        return recognizer.dominantLanguage
    }

    private func sentenceSplit(_ text: String, language: NLLanguage?) -> [String] {
        let tokenizer = NLTokenizer(unit: .sentence)
        if let language { tokenizer.setLanguage(language) }
        tokenizer.string = text
        var result: [String] = []
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let sentence = text[range].trimmingCharacters(in: .whitespacesAndNewlines)
            if !sentence.isEmpty { result.append(sentence) }
            return true
        }
        return result.isEmpty ? [text] : result
    }

    /// Fold early-modern typography toward modern spelling: long-s (ſ) → s, common
    /// typographic ligatures expanded, then NFC. Intentionally conservative — it leaves
    /// u/v and i/j alone (modernizing those is lossy and period/language-specific).
    private func normalizeOrthography(_ s: String) -> String {
        var out = s.replacingOccurrences(of: "\u{017F}", with: "s")  // ſ long s
        let ligatures: [Character: String] = [
            "\u{FB00}": "ff", "\u{FB01}": "fi", "\u{FB02}": "fl",
            "\u{FB03}": "ffi", "\u{FB04}": "ffl", "\u{FB05}": "st", "\u{FB06}": "st",
        ]
        let mapped: [Character] = out.flatMap { ch -> [Character] in
            if let rep = ligatures[ch] { return Array(rep) }
            return [ch]
        }
        out = String(mapped)
        return out.precomposedStringWithCanonicalMapping  // NFC
    }
}

// MARK: - Internal model

/// A normalized piece of a block, before stitching.
private enum Candidate {
    case prose(text: String, label: String, continuable: Bool, preformatted: Bool)
    case heading(level: Int, text: String)
    case list(ordered: Bool, items: [String])
    case table(html: String)
    case equation(html: String)
    case figure(caption: String?)
}

/// A paragraph being accumulated across stitched blocks.
private struct ProseAccumulator {
    var text: String
    var label: String
    var continuable: Bool
}

// MARK: - Sentence-boundary helpers

private let sentenceTerminators: Set<Character> = [
    ".", "!", "?", "\u{2026}",  // . ! ? …
    "\u{3002}", "\u{FF01}", "\u{FF1F}",  // 。！？ (CJK)
    "\u{061F}", "\u{06D4}",  // ؟ ۔ (Arabic/Urdu)
    "\u{0964}", "\u{0965}",  // । ॥ (Devanagari danda)
]

private let closingPunctuation: Set<Character> = [
    "\"", "'", "\u{201D}", "\u{2019}", "\u{00BB}", "\u{203A}",
    ")", "]", "}", "\u{300D}", "\u{300F}", "\u{FF09}",
]

/// True if the last meaningful character (skipping trailing whitespace and closing
/// quotes/brackets) is sentence-terminal punctuation.
private func endsWithSentenceTerminator(_ s: String) -> Bool {
    var idx = s.endIndex
    while idx > s.startIndex {
        let prev = s.index(before: idx)
        let ch = s[prev]
        if ch.isWhitespace || closingPunctuation.contains(ch) {
            idx = prev
            continue
        }
        return sentenceTerminators.contains(ch)
    }
    return false
}

// MARK: - Trimming helpers

extension String {
    /// Self with trailing whitespace/newlines removed.
    fileprivate func trimmedTrailingWhitespace() -> String {
        var end = endIndex
        while end > startIndex {
            let prev = index(before: end)
            if self[prev].isWhitespace { end = prev } else { break }
        }
        return String(self[startIndex..<end])
    }

    /// Self with leading whitespace/newlines removed.
    fileprivate func trimmedLeadingWhitespace() -> String {
        var start = startIndex
        while start < endIndex, self[start].isWhitespace { start = index(after: start) }
        return String(self[start..<endIndex])
    }
}

// MARK: - Minimal HTML extraction

/// A small, dependency-free extractor for surya's constrained HTML tag set. Strips tags,
/// decodes entities, and recovers paragraph/list/heading structure — enough for prose
/// without pulling in a full HTML parser.
private enum HTMLText {
    /// Plain text of an HTML fragment: `<br>` and block-close tags become separators, all
    /// tags are removed, entities decoded, and whitespace collapsed. With `preserveBreaks`,
    /// line breaks are kept (for `<pre>`/code).
    static func plainText(_ html: String, preserveBreaks: Bool = false) -> String {
        var t = html
        t = replacingRegex(t, "(?i)<br\\s*/?>", with: preserveBreaks ? "\n" : " ")
        t = replacingRegex(t, "(?i)</(p|div|li|h[1-6]|tr|td|th)>", with: " ")
        t = replacingRegex(t, "<[^>]+>", with: "")
        t = decodeEntities(t)
        if preserveBreaks {
            t = replacingRegex(t, "[ \\t]+", with: " ")
            return t.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        t = replacingRegex(t, "\\s+", with: " ")
        return t.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Split a block's HTML into one-or-more plain-text paragraphs (on `</p>` and double `<br>`).
    static func paragraphs(_ html: String) -> [String] {
        var t = replacingRegex(html, "(?i)<br\\s*/?>\\s*<br\\s*/?>", with: "</p>")
        let parts = splittingRegex(t, "(?i)</p>").map { plainText($0) }.filter { !$0.isEmpty }
        if !parts.isEmpty { return parts }
        let whole = plainText(html)
        return whole.isEmpty ? [] : [whole]
    }

    /// Plain text of each `<li>` item.
    static func listItems(_ html: String) -> [String] {
        capturing(html, "(?is)<li[^>]*>(.*?)</li>").map { plainText($0) }.filter { !$0.isEmpty }
    }

    /// Heading level from the first `<h1>`…`<h6>`, if present.
    static func headingLevel(_ html: String) -> Int? {
        capturing(html, "(?i)<h([1-6])").first.flatMap(Int.init)
    }

    // Entity decoding ---------------------------------------------------------

    private static let namedEntities: [String: Character] = [
        "amp": "&", "lt": "<", "gt": ">", "quot": "\"", "apos": "'", "nbsp": " ",
        "mdash": "\u{2014}", "ndash": "\u{2013}", "hellip": "\u{2026}",
        "laquo": "\u{00AB}", "raquo": "\u{00BB}", "lsquo": "\u{2018}", "rsquo": "\u{2019}",
        "ldquo": "\u{201C}", "rdquo": "\u{201D}", "deg": "\u{00B0}", "middot": "\u{00B7}",
    ]

    private static func decodeEntities(_ s: String) -> String {
        guard s.contains("&") else { return s }
        var result = ""
        result.reserveCapacity(s.count)
        var i = s.startIndex
        while i < s.endIndex {
            let ch = s[i]
            if ch == "&", let semi = s[i...].firstIndex(of: ";"),
                s.distance(from: i, to: semi) <= 12
            {
                let body = String(s[s.index(after: i)..<semi])
                if let decoded = decodeOne(body) {
                    result.append(decoded)
                    i = s.index(after: semi)
                    continue
                }
            }
            result.append(ch)
            i = s.index(after: i)
        }
        return result
    }

    private static func decodeOne(_ body: String) -> Character? {
        if let named = namedEntities[body] { return named }
        guard body.hasPrefix("#") else { return nil }
        let digits = body.dropFirst()
        let scalarValue: UInt32?
        if digits.hasPrefix("x") || digits.hasPrefix("X") {
            scalarValue = UInt32(digits.dropFirst(), radix: 16)
        } else {
            scalarValue = UInt32(digits, radix: 10)
        }
        guard let value = scalarValue, let scalar = Unicode.Scalar(value) else { return nil }
        return Character(scalar)
    }

    // Regex helpers -----------------------------------------------------------

    private static func replacingRegex(_ s: String, _ pattern: String, with template: String) -> String {
        guard let re = try? NSRegularExpression(pattern: pattern) else { return s }
        let range = NSRange(s.startIndex..., in: s)
        return re.stringByReplacingMatches(in: s, range: range, withTemplate: template)
    }

    private static func splittingRegex(_ s: String, _ pattern: String) -> [String] {
        guard let re = try? NSRegularExpression(pattern: pattern) else { return [s] }
        let ns = s as NSString
        var result: [String] = []
        var last = 0
        re.enumerateMatches(in: s, range: NSRange(location: 0, length: ns.length)) { match, _, _ in
            guard let match else { return }
            result.append(ns.substring(with: NSRange(location: last, length: match.range.location - last)))
            last = match.range.location + match.range.length
        }
        result.append(ns.substring(from: last))
        return result
    }

    private static func capturing(_ s: String, _ pattern: String) -> [String] {
        guard let re = try? NSRegularExpression(pattern: pattern) else { return [] }
        let ns = s as NSString
        return re.matches(in: s, range: NSRange(location: 0, length: ns.length)).compactMap { match in
            guard match.numberOfRanges > 1 else { return nil }
            let r = match.range(at: 1)
            return r.location == NSNotFound ? nil : ns.substring(with: r)
        }
    }
}

// MARK: - Codable

extension DocElement: Codable {
    private enum CodingKeys: String, CodingKey {
        case type, level, text, paragraph, ordered, items, html, caption
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .heading(let level, let text):
            try c.encode("heading", forKey: .type)
            try c.encode(level, forKey: .level)
            try c.encode(text, forKey: .text)
        case .paragraph(let paragraph):
            try c.encode("paragraph", forKey: .type)
            try c.encode(paragraph, forKey: .paragraph)
        case .list(let ordered, let items):
            try c.encode("list", forKey: .type)
            try c.encode(ordered, forKey: .ordered)
            try c.encode(items, forKey: .items)
        case .table(let html):
            try c.encode("table", forKey: .type)
            try c.encode(html, forKey: .html)
        case .equation(let html):
            try c.encode("equation", forKey: .type)
            try c.encode(html, forKey: .html)
        case .figure(let caption):
            try c.encode("figure", forKey: .type)
            try c.encodeIfPresent(caption, forKey: .caption)
        }
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        switch try c.decode(String.self, forKey: .type) {
        case "heading":
            self = .heading(
                level: try c.decode(Int.self, forKey: .level),
                text: try c.decode(String.self, forKey: .text))
        case "paragraph":
            self = .paragraph(try c.decode(Paragraph.self, forKey: .paragraph))
        case "list":
            self = .list(
                ordered: try c.decode(Bool.self, forKey: .ordered),
                items: try c.decode([String].self, forKey: .items))
        case "table":
            self = .table(html: try c.decode(String.self, forKey: .html))
        case "equation":
            self = .equation(html: try c.decode(String.self, forKey: .html))
        case "figure":
            self = .figure(caption: try c.decodeIfPresent(String.self, forKey: .caption))
        case let other:
            throw DecodingError.dataCorruptedError(
                forKey: .type, in: c, debugDescription: "unknown DocElement type '\(other)'")
        }
    }
}
