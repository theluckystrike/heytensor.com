from pathlib import Path
import json
import re

root = Path(__file__).resolve().parent.parent
calculator = (root / "assets/js/components/shape-calculator.js").read_text(encoding="utf-8-sig")
data = json.loads((root / "data/_tools.json").read_text(encoding="utf-8-sig"))
page = (root / "tools/multihead-attention-calculator.html").read_text(encoding="utf-8-sig")

assert re.search(r"Attention weights \(default, averaged over heads\):", calculator), "calculator missing default averaged attention weights shape"
assert re.search(r"With average_attn_weights=False:", calculator), "calculator missing per-head attention weights shape label"
faq = next(tool["faq"] for tool in data if tool.get("config", {}).get("layer") == "MultiheadAttention")
faq_text = faq[0]["a"]
assert "[batch, seq_len, seq_len] (default, averaged over heads)" in faq_text, "tool FAQ missing default averaged attention weights shape"
assert "[batch, num_heads, seq_len, seq_len] with average_attn_weights=False" in faq_text, "tool FAQ missing explicit per-head attention weights shape"
assert page.count("[batch, seq_len, seq_len] (default, averaged over heads)") == 2, "built page FAQs missing default averaged attention weights shape"
assert page.count("[batch, num_heads, seq_len, seq_len] with average_attn_weights=False") == 2, "built page FAQs missing explicit per-head attention weights shape"
print("MultiheadAttention averaged and per-head weight shapes are documented in the calculator and both FAQ copies.")
