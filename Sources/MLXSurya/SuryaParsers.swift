import Foundation

/// One layout block parsed from the model's JSON output (bbox in pixels).
public struct ParsedLayoutBlock: Sendable, Equatable {
    public let label: String
    public let bbox: [Double]
    public let count: Int
}

/// One table row/column parsed from the model's JSON output (bbox in pixels).
public struct ParsedTableElement: Sendable, Equatable {
    public let label: String  // "Row" or "Col"
    public let bbox: [Double]
}

/// One block parsed from full-page HTML output (bbox in pixels, inner HTML cleaned).
public struct ParsedFullPageBlock: Sendable, Equatable {
    public let label: String
    public let bbox: [Double]
    public let html: String
}

/// Pure-Swift port of `surya/inference/parsers.py`. Turns raw model text (JSON or HTML) into
/// typed blocks with pixel-space bboxes. No model dependency — directly unit-testable.
public enum SuryaParsers {

    /// Denormalize a `[x0, y0, x1, y1]` bbox from the model's 0..`scale` space to pixels.
    /// Ports `denorm_bbox` (`BBOX_SCALE` = 1000).
    public static func denormBbox(
        _ bbox: [Double], imageWidth: Int, imageHeight: Int, scale: Int = 1000
    ) -> [Double] {
        guard bbox.count == 4 else { return [0, 0, 0, 0] }
        let s = Double(scale)
        let w = Double(imageWidth)
        let h = Double(imageHeight)
        return [bbox[0] / s * w, bbox[1] / s * h, bbox[2] / s * w, bbox[3] / s * h]
    }

    // MARK: - Layout (JSON)

    /// Port of `parse_layout`. Returns blocks with pixel-space bboxes.
    public static func parseLayout(
        _ text: String, imageWidth: Int, imageHeight: Int, scale: Int = 1000
    ) -> [ParsedLayoutBlock] {
        guard let arr = extractJSONArray(text) else { return [] }
        var blocks: [ParsedLayoutBlock] = []
        for case let obj as [String: Any] in arr {
            guard let label = obj["label"] as? String else { continue }
            guard let norm = coerceBbox(obj["bbox"]) else { continue }
            let count = coerceInt(obj["count"]) ?? 0
            blocks.append(
                ParsedLayoutBlock(
                    label: label,
                    bbox: denormBbox(norm, imageWidth: imageWidth, imageHeight: imageHeight, scale: scale),
                    count: count))
        }
        return blocks
    }

    // MARK: - Table recognition (JSON)

    /// Port of `parse_table_rec`. Keeps only `Row`/`Col` entries; pixel-space bboxes.
    public static func parseTableRec(
        _ text: String, imageWidth: Int, imageHeight: Int, scale: Int = 1000
    ) -> [ParsedTableElement] {
        guard let arr = extractJSONArray(text) else { return [] }
        var els: [ParsedTableElement] = []
        for case let obj as [String: Any] in arr {
            guard let label = obj["label"] as? String, label == "Row" || label == "Col" else {
                continue
            }
            guard let norm = coerceBbox(obj["bbox"]) else { continue }
            els.append(
                ParsedTableElement(
                    label: label,
                    bbox: denormBbox(norm, imageWidth: imageWidth, imageHeight: imageHeight, scale: scale)))
        }
        return els
    }

    // MARK: - Block HTML

    /// Port of `clean_block_html`: strip markdown code fences, trim.
    public static func cleanBlockHTML(_ text: String) -> String {
        stripCodeFences(text).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Full-page HTML

    /// Port of `parse_full_page_html`. Scans **top-level** `<div>` elements, reads their
    /// `data-bbox` / `data-label`, denormalizes the bbox to pixels, and returns the inner HTML
    /// with the debug `data-bbox` / `data-label` attributes stripped from children.
    public static func parseFullPageHTML(
        _ text: String, imageWidth: Int, imageHeight: Int, scale: Int = 1000
    ) -> [ParsedFullPageBlock] {
        let html = stripCodeFences(text)
        var result: [ParsedFullPageBlock] = []
        var searchStart = html.startIndex

        while let openStart = html.range(
            of: "<div", options: .caseInsensitive, range: searchStart..<html.endIndex)
        {
            guard
                let openTagEnd = html.range(
                    of: ">", range: openStart.upperBound..<html.endIndex)
            else { break }
            let openTag = String(html[openStart.lowerBound..<openTagEnd.upperBound])

            // Walk forward to the matching </div>, tracking nested-div depth.
            var depth = 1
            var cursor = openTagEnd.upperBound
            var contentEnd: String.Index?
            var afterClose: String.Index = html.endIndex
            while depth > 0 {
                let nextOpen = html.range(
                    of: "<div", options: .caseInsensitive, range: cursor..<html.endIndex)
                guard
                    let nextClose = html.range(
                        of: "</div", options: .caseInsensitive, range: cursor..<html.endIndex)
                else { break }
                if let open = nextOpen, open.lowerBound < nextClose.lowerBound {
                    depth += 1
                    cursor = open.upperBound
                } else {
                    depth -= 1
                    cursor = nextClose.upperBound
                    if depth == 0 {
                        contentEnd = nextClose.lowerBound
                        afterClose =
                            html.range(of: ">", range: nextClose.upperBound..<html.endIndex)?
                            .upperBound ?? html.endIndex
                    }
                }
            }

            guard let cEnd = contentEnd else {
                searchStart = openTagEnd.upperBound
                continue
            }
            let inner = String(html[openTagEnd.upperBound..<cEnd])
            let label = attribute("data-label", in: openTag) ?? ""
            let norm = attribute("data-bbox", in: openTag).flatMap(parseBboxString) ?? [0, 0, 0, 0]
            let bbox = denormBbox(norm, imageWidth: imageWidth, imageHeight: imageHeight, scale: scale)
            let cleanedInner = stripDebugAttributes(inner)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            result.append(ParsedFullPageBlock(label: label, bbox: bbox, html: cleanedInner))
            searchStart = afterClose
        }
        return result
    }

    // MARK: - Helpers

    /// Strip leading/trailing markdown code fences (```json … ```), keeping inner content.
    static func stripCodeFences(_ text: String) -> String {
        var s = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if s.hasPrefix("```") {
            if let firstNewline = s.firstIndex(of: "\n") {
                s = String(s[s.index(after: firstNewline)...])  // drop ```json line
            } else {
                s = String(s.dropFirst(3))
            }
        }
        if s.hasSuffix("```") {
            s = String(s.dropLast(3))
        }
        return s.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Extract the first `[ … ]` JSON array (greedy, spanning newlines) and parse it.
    static func extractJSONArray(_ text: String) -> [Any]? {
        let s = stripCodeFences(text)
        guard
            let re = try? NSRegularExpression(
                pattern: "\\[.*\\]", options: [.dotMatchesLineSeparators])
        else { return nil }
        let ns = s as NSString
        guard let m = re.firstMatch(in: s, range: NSRange(location: 0, length: ns.length))
        else { return nil }
        let json = ns.substring(with: m.range)
        guard let data = json.data(using: .utf8),
            let obj = try? JSONSerialization.jsonObject(with: data) as? [Any]
        else { return nil }
        return obj
    }

    /// Coerce a model `bbox` field (space-separated string or numeric array) to `[Double]` of 4.
    static func coerceBbox(_ value: Any?) -> [Double]? {
        if let str = value as? String {
            return parseBboxString(str)
        }
        if let arr = value as? [Any] {
            let nums = arr.compactMap { coerceDouble($0) }
            return nums.count == 4 ? nums : nil
        }
        return nil
    }

    /// Parse `"x0 y0 x1 y1"` into 4 Doubles (nil unless exactly 4 numbers).
    static func parseBboxString(_ s: String) -> [Double]? {
        let parts = s.split(whereSeparator: { $0 == " " || $0 == "," })
            .compactMap { Double($0) }
        return parts.count == 4 ? parts : nil
    }

    static func coerceInt(_ value: Any?) -> Int? {
        if let i = value as? Int { return i }
        if let d = value as? Double { return Int(d) }
        if let s = value as? String { return Int(s) ?? Double(s).map(Int.init) }
        return nil
    }

    static func coerceDouble(_ value: Any) -> Double? {
        if let d = value as? Double { return d }
        if let i = value as? Int { return Double(i) }
        if let s = value as? String { return Double(s) }
        return nil
    }

    /// Read an attribute value (`name="…"` or `name='…'`) from an opening tag string.
    static func attribute(_ name: String, in tag: String) -> String? {
        let pattern = "\(NSRegularExpression.escapedPattern(for: name))\\s*=\\s*[\"']([^\"']*)[\"']"
        guard let re = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive),
            let m = re.firstMatch(
                in: tag, range: NSRange(tag.startIndex..., in: tag)),
            let r = Range(m.range(at: 1), in: tag)
        else { return nil }
        return String(tag[r])
    }

    /// Strip `data-bbox="…"` and `data-label="…"` attributes (the model's debug markup) from
    /// inner HTML.
    static func stripDebugAttributes(_ html: String) -> String {
        var out = html
        for name in ["data-bbox", "data-label"] {
            let pattern = "\\s*\(name)\\s*=\\s*[\"'][^\"']*[\"']"
            if let re = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive) {
                out = re.stringByReplacingMatches(
                    in: out, range: NSRange(out.startIndex..., in: out), withTemplate: "")
            }
        }
        return out
    }
}
