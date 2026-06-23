"""VLM input reference: render surya-ocr-2's chat template (jinja2) + encode (Rust tokenizers).

AutoTokenizer can't load this model's custom `TokenizersBackend` class, so we use the raw
tokenizer.json + chat_template.jinja directly — the same two artifacts the Swift SuryaTokenizer
reimplements. Output: the rendered prompt + reference input_ids (single <|image_pad|>, pre-grid).
"""
import json
import sys
import jinja2
from tokenizers import Tokenizer

SNAP, OUT = sys.argv[1], sys.argv[2]
PROMPT = "OCR this image to HTML. Each block is a div with data-label and data-bbox (x0 y0 x1 y1, normalized 0-1000)."

tok = Tokenizer.from_file(f"{SNAP}/tokenizer.json")
template_str = open(f"{SNAP}/chat_template.jinja").read()

def raise_exception(msg):
    raise Exception(msg)

env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
env.globals["raise_exception"] = raise_exception
template = env.from_string(template_str)

msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
rendered = template.render(messages=msgs, add_generation_prompt=True)
ids = tok.encode(rendered, add_special_tokens=False).ids

json.dump({"rendered": rendered, "input_ids": ids}, open(OUT, "w"))
print("rendered chars:", len(rendered), "| n_ids:", len(ids))
print("first12:", ids[:12], "last6:", ids[-6:])
print(repr(rendered[:120]))
