"""Port the worst-case proof of a graph to its average-case package.

The average-case package holds the worst-case export renamed (`worst/W_*.lean`, see
`gen_bridge.py`); this script copies the worst-case proof modules into `proof/` with the same
renaming (`Stage_x` → `W_Stage_x`, `Ideal.` → `W_Ideal.`, `Requirement_0` → `W_Requirement_0`,
`GeneratedClaim.Runs` and the stage parameters → `W_GeneratedClaim.…`, `import Claim` →
`import W_Claim`), so the worst-case correctness theorem is stated over the renamed export.

Usage: `python3 gen_port.py <average directory> <worst directory> <module> ...`.
"""
import os
import re
import sys

avg, worst, modules = sys.argv[1], sys.argv[2], sys.argv[3:]
os.makedirs(f'{avg}/proof', exist_ok=True)


def port(text):
    text = re.sub(r'\bStage_(\w+)', r'W_Stage_\1', text)
    text = re.sub(r'\bIdeal\.', 'W_Ideal.', text)
    text = re.sub(r'\bRequirement_0\b', 'W_Requirement_0', text)
    text = re.sub(r'\bGeneratedClaim\.(Runs|stage_\d+_params|requirement_0_params|ideal_params)\b',
                  r'W_GeneratedClaim.\1', text)
    text = re.sub(r'^import Claim$', 'import W_Claim', text, flags=re.M)
    text = re.sub(r'^import (Ideal|Requirement_0)$', r'import W_\1', text, flags=re.M)
    return text


for m in modules:
    open(f'{avg}/proof/{m}.lean', 'w').write(port(open(f'{worst}/{m}.lean').read()))
