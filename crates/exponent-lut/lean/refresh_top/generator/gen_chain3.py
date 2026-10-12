"""Adapt the proof files of a chain-3 part (`3_j`) to its decoding ring `D3 = q₀ q₁ q₃`.

The generators and the shared lemmas are written for the ring `Q₁ = q₀ q₁ q₂` of chains 0–2.
This script
- writes `Part_3_j_Local.lean`: the lemmas of the `Top*` files that mention `Q₁` (or its
  constants), with `Q₁` replaced by `D3`, in the part's namespace, where they shadow the shared
  ones;
- rewrites the part's files: `Q₁` → `D3` and its constants (layout, inverses, half), and imports
  the local lemmas.

Usage: `python3 gen_chain3.py <parameter directory> <part>`.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from chains import d3_subst

TOP = ['TopRefresh', 'TopAction', 'TopColumn', 'TopHint', 'TopTarget', 'TopGlue', 'TopChain']
MARK = re.compile(r'\bQ1\b|\bhalfQ1\b|\bQqPoly\b|\blayoutQ1\b')
SKIP = {'actionConst', 'QqPoly', 'actionConst_mul', 'action_mul', 'actionValue', 'column_value',
        'decompose_rows', 'row_mul_qq', 'refresh_column', 'column_noise', 'six_rows', 'fresh_glue',
        'y_unique', 'actionValue_eq'}




def chunks(text):
    """The top-level declarations of a file body, each with its doc comment and options."""
    body = text[text.index('namespace RefreshTopProof'):text.rindex('end RefreshTopProof')]
    body = body[body.index('\n') + 1:]
    out, cur = [], []
    for line in body.split('\n'):
        if re.match(r'(end |namespace |open |import |section|variable)', line):
            continue
        if re.match(r'(/--|set_option|theorem|noncomputable def|def|abbrev|lemma)', line) and cur and \
                not cur[-1].startswith(('/--', 'set_option')) and not (cur[-1].strip().endswith('-/') and
                                                                     any(c.startswith('/--') for c in cur[-3:])):
            if any(re.match(r'(theorem|noncomputable def|def|abbrev|lemma) ', c) for c in cur):
                out.append('\n'.join(cur))
                cur = []
        cur.append(line)
    if cur:
        out.append('\n'.join(cur))
    return out


def name_of(chunk):
    m = re.search(r'^(?:theorem|noncomputable def|def|abbrev|lemma) (\S+)', chunk, re.M)
    return m.group(1) if m else None


if __name__ == '__main__':
    d, part = sys.argv[1], sys.argv[2]
    subst = lambda text: d3_subst(text, d)
    local = [f'''import TopChain
    import TopLinkC
    import TopFinal

    /-! The lemmas of the decoding ring for chain-3 part `{part}`: those of the shared `Top*` files
    that mention `Q₁`, at `D3` (written by `generator/gen_chain3.py`). -/

    namespace RefreshTopProof.Part_{part}

    open Mxx.Primitives MxxRuntime MxxExponentLut Mxx.ExponentLut.Refresh RefreshTopProof
    ''']
    for f in TOP:
        for c in chunks(open(f'{d}/{f}.lean').read()):
            n = name_of(c)
            if n is None or n in SKIP or not MARK.search(c):
                continue
            local.append(subst(c).rstrip() + '\n')
    local.append(f'end RefreshTopProof.Part_{part}\n')
    open(f'{d}/Part_{part}_Local.lean', 'w').write('\n'.join(local))

    for fn in sorted(os.listdir(d)):
        if not re.fullmatch(rf'Part_{part}_\w+\.lean', fn) or fn.endswith(('_Local.lean', '_Assembly.lean')):
            continue
        text = open(f'{d}/{fn}').read()
        if 'import Part_' + part + '_Local' not in text:
            text = re.sub(r'^(import [^\n]*\n)', r'\1import Part_' + part + '_Local\n', text, count=1)
        text = subst(text)
        if not fn.endswith('_OnlineFinal.lean'):
            # The scale of the level above the ring: `Q₀ = q₂ D3` (the basis lists keep `q₃`).
            text = re.sub(r'\[q0, q1, q2, q3', '[@BASIS@', text)
            text = re.sub(r'\bq3\b', 'q2', text)
            text = text.replace('[@BASIS@', '[q0, q1, q2, q3')
        open(f'{d}/{fn}', 'w').write(text)
