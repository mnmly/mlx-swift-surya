import Foundation

/// Which task prompt to send to the surya-ocr-2 VLM. Mirrors the `PROMPT_TYPE_*`
/// constants in `surya/inference/prompts.py`.
public enum SuryaPromptType: String, Sendable, CaseIterable {
    /// Layout analysis → JSON array of `{label, bbox, count}` (`PROMPT_TYPE_LAYOUT`).
    case layout
    /// Per-block OCR → HTML (`PROMPT_TYPE_BLOCK`).
    case block
    /// Table structure → JSON array of `{label: Row|Col, bbox}` (`PROMPT_TYPE_TABLE_REC`).
    case tableRec = "table_rec"
    /// Full-page OCR → top-level `<div data-bbox data-label>` blocks
    /// (`PROMPT_TYPE_HIGH_ACCURACY_BBOX`).
    case highAccuracyBbox = "high_accuracy_bbox"
}

/// The exact prompt strings and label vocabularies from `surya/inference/prompts.py`
/// and `surya/layout/label.py`.
public enum SuryaPrompts {
    /// The prompt text sent for a given task. Verbatim from `surya/inference/prompts.py`.
    public static func prompt(for type: SuryaPromptType) -> String {
        switch type {
        case .layout:
            return
                "Output the layout of this image as JSON. Each entry is a dict with \"label\", \"bbox\", and \"count\" fields. Bbox is x0 y0 x1 y1, normalized 0-1000."
        case .block:
            return "OCR this block image to HTML."
        case .tableRec:
            return
                "Output the table rows then columns as JSON. Each entry is a dict with \"label\" (\"Row\" or \"Col\") and \"bbox\" (x0 y0 x1 y1, normalized 0-1000)."
        case .highAccuracyBbox:
            return
                "OCR this image to HTML. Each block is a div with data-label and data-bbox (x0 y0 x1 y1, normalized 0-1000)."
        }
    }

    /// The 19 canonical layout labels the model emits (`LAYOUT_LABEL_SET`).
    public static let layoutLabels: [String] = [
        "Caption", "Footnote", "Equation-Block", "List-Group", "Page-Header",
        "Page-Footer", "Image", "Section-Header", "Table", "Text",
        "Complex-Block", "Code-Block", "Form", "Table-Of-Contents", "Figure",
        "Chemical-Block", "Diagram", "Bibliography", "Blank-Page",
    ]

    /// Raw model label → canonical surya label (`LAYOUT_PRED_RELABEL`).
    public static let layoutRelabel: [String: String] = [
        "Caption": "Caption",
        "Footnote": "Footnote",
        "Equation-Block": "Equation",
        "List-Group": "ListGroup",
        "Page-Header": "PageHeader",
        "Page-Footer": "PageFooter",
        "Image": "Picture",
        "Section-Header": "SectionHeader",
        "Table": "Table",
        "Text": "Text",
        "Complex-Block": "Figure",
        "Code-Block": "Code",
        "Form": "Form",
        "Table-Of-Contents": "TableOfContents",
        "Figure": "Figure",
        "Chemical-Block": "ChemicalBlock",
        "Diagram": "Diagram",
        "Bibliography": "Bibliography",
        "Blank-Page": "BlankPage",
    ]

    /// Canonicalize a raw model label; unknown labels pass through unchanged.
    public static func canonicalLabel(_ raw: String) -> String {
        layoutRelabel[raw] ?? raw
    }

    /// Text-bearing canonical labels eligible for blank-region filtering (`TEXT_LABELS`).
    public static let textLabels: Set<String> = [
        "Text", "SectionHeader", "PageHeader", "PageFooter",
        "Caption", "Footnote", "Code", "Bibliography",
    ]

    /// Raw model labels that are never OCR'd (`SKIP_OCR_LABELS`).
    public static let skipOCRLabels: Set<String> = ["Figure", "Image", "Diagram", "Blank-Page"]

    /// Canonicalized form of ``skipOCRLabels`` (what block-mode recognition compares against).
    public static let skipCanonLabels: Set<String> = Set(skipOCRLabels.map(canonicalLabel))

    /// JSON schema enforced for layout when guided decoding is enabled (`LAYOUT_JSON_SCHEMA`).
    /// Retained for documentation + future guided-decoding support; the greedy decode path does
    /// not currently constrain output (the model is trained to emit this shape).
    public static let layoutJSONSchema = """
        {"type":"array","maxItems":200,"items":{"type":"object","properties":\
        {"label":{"type":"string","enum":[...19 labels...]},\
        "bbox":{"type":"string","pattern":"^\\\\d{1,4} \\\\d{1,4} \\\\d{1,4} \\\\d{1,4}$"},\
        "count":{"type":"integer","minimum":0,"maximum":10000}},\
        "required":["label","bbox","count"],"additionalProperties":false}}
        """
}
