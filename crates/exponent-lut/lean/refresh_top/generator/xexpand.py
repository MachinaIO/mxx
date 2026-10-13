#!/usr/bin/env python3
"""The generators' view of an export: every exported module with its abbreviations expanded.

The export names its long decimal literals (`mxxNat<k>`) and its matrix types
`Mxx.Primitives.ExactMatrix q n r c` (`mxxMat<k>`, whose arguments may name literals) by blocks of
local notations, each followed by a blank line (see `abbreviate_long_literals` in
`crates/ir-core/src/lean/mod.rs`). A notation expands to the text it replaced, so substituting
every name back, latest declaration first, and dropping the blocks restores the unabbreviated
module text, which the generators recognize. The modules are built as exported; the generated
proofs elaborate to the same terms either way.

Usage: xexpand.py <export dir> <view dir>
"""
import os
import re
import sys

DECL = re.compile(r'^local notation "(mxx(?:Nat|Mat)\d+)" => (.*)\n', re.M)
BLOCK = re.compile(r'^(?:local notation "mxx(?:Nat|Mat)\d+" => .*\n)+\n', re.M)


def expand(text):
    """`text` with every abbreviation of its `local notation` declarations expanded."""
    decls = list(DECL.finditer(text))
    body = BLOCK.sub('', text)
    for m in reversed(decls):
        body = re.sub(rf'\b{m.group(1)}\b', lambda _: m.group(2), body)
    return body


def main(src, dst):
    os.makedirs(dst, exist_ok=True)
    for name in sorted(os.listdir(src)):
        if name.endswith('.lean'):
            with open(os.path.join(src, name)) as f:
                text = expand(f.read())
            with open(os.path.join(dst, name), 'w') as f:
                f.write(text)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
