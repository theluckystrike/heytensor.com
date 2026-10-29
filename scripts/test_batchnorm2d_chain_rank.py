from pathlib import Path
import re
import sys

root = Path(__file__).resolve().parent.parent
source = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "assets/js/components/shape-calculator.js"
calculator = source.read_text(encoding="utf-8-sig")

branch = re.search(r"case 'BatchNorm2d':(?P<body>.*?)break;", calculator, re.S)
assert branch, "BatchNorm2d chain input mapping is missing"
assert re.search(
    r"if\s*\(shape\.length\s*!==\s*4\)\s*return\s*'BatchNorm2d expects a 4D input, got '\s*\+\s*shape\.length\s*\+\s*'D\.';",
    branch.group("body"),
), "BatchNorm2d must reject non-4D chain inputs with a rank-specific error"
assert re.search(
    r"p\.batch\s*=\s*shape\[0\].*?p\.num_features\s*=\s*shape\[1\]",
    branch.group("body"),
    re.S,
), "BatchNorm2d must map all four dimensions for valid inputs"
print("BatchNorm2d chain inputs reject non-4D shapes and map valid 4D shapes.")
