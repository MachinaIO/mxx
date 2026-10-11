"""Chain theorems: one theorem over the atoms of several segment theorems of a root, applying them
in sequence (see `segment.py`).

`seg_args(text, thm)` gives the explicit atom and `let` binders of segment theorem `thm` in Lean
source `text`; `seg_call(text, thm)` passes each of them by name, so a chain theorem whose
binders come from `segment` over the union of the atoms applies `thm` without restating them.
"""
import re

from segment import segment

WIRE = re.compile(r'\bw_\d+_(?:\d+|concat_\d+)\b')


def head(text, thm):
    i = text.index(f'theorem {thm} ')
    return text[i:text.index(':= by', i)]


def seg_args(text, thm):
    return re.findall(r'\((h\d\w*|e_\w+) :', head(text, thm))


def seg_call(text, thm):
    return f'{thm} (model := model) ' + ' '.join(f'({b} := {b})' for b in seg_args(text, thm))


def chain_binders(path, ns, texts, ab, extra_atoms=(), extra_wires=()):
    """The binders of a chain over the segment theorems `texts` (pairs `(text, thm)`) and the
    atoms `extra_atoms`."""
    atoms, wires = list(extra_atoms), list(extra_wires)
    for text, thm in texts:
        for b in seg_args(text, thm):
            if b.startswith('e_'):
                wires.append('w_' + b[2:])
            elif b not in atoms:
                atoms.append(b)
    binders, _ = segment(path, ns, 'generatedRoot', atoms, ab, wires)
    return binders


def skip_arg(s, i):
    """The index after the balanced argument of `s` starting at `i` (after spaces)."""
    while s[i].isspace():
        i += 1
    depth = 0
    j = i
    while True:
        ch = s[j]
        if ch in '([{⟨':
            depth += 1
        elif ch in ')]}⟩':
            depth -= 1
        if depth == 0 and (ch in ')]}⟩' or s[j + 1].isspace()):
            return j + 1
        j += 1


def conclusion_bound(text, thm):
    """The bound (last argument) of the `Rel` conclusion of `thm`."""
    h = head(text, thm)
    i = h.rindex('\n    Rel sm ') + len('\n    Rel sm ')
    for _ in range(3):
        i = skip_arg(h, i)
    return ' '.join(h[i:].split())


def subst(expr, mapping):
    return re.sub(r"\b(\w+'?)(?![\w'])", lambda m: mapping.get(m.group(1), m.group(1)), expr)


def binder_groups(text, thm):
    """The binder groups (`{...}`, `(...)`, `[...]`) of theorem `thm` and its conclusion."""
    h = head(text, thm)
    i = h.index(f'theorem {thm} ') + len(f'theorem {thm} ')
    groups = []
    while True:
        while h[i].isspace():
            i += 1
        if h[i] == ':':
            return groups, h[i + 1:].strip()
        j = skip_arg(h, i)
        groups.append(h[i:j])
        i = j


def group_names(g):
    """The names a binder group binds."""
    return g[1:g.index(' :')].split()


def wrap(text, thm, mapping, atom_args):
    """Specialize theorem `thm` to root wires: the binder groups binding only names of `mapping`
    (generic wires) or of `atom_args` (hypotheses supplied by root atoms) are dropped, the rest
    renamed by `mapping`. Returns `(groups, conclusion, call)`, `call` applying `thm` with the
    atoms and every kept explicit binder by name."""
    groups, concl = binder_groups(text, thm)
    kept, explicit = [], []
    for g in groups:
        names = group_names(g)
        if all(n in mapping or n in atom_args or n in ('model', 'params') for n in names):
            continue
        kept.append(subst(g, mapping))
        if g.startswith('('):
            explicit += names
    call = f'{thm} (model := model) ' + ' '.join(f'({k} := {v})' for k, v in atom_args.items()) + \
        ' ' + ' '.join(f'({n} := {n})' for n in explicit)
    return kept, subst(concl, mapping), call


def prefixed(text, p='p'):
    """Rename the wires (`w_…` → `pw_…`), atoms (`h…` → `ph…`) and `let` equations (`e_…` →
    `pe_…`) of a root, so two roots can share a context."""
    text = re.sub(r'(?<![\w.])(w_\d+_(?:\d+|concat_\d+))\b', p + r'\1', text)
    text = re.sub(r'(?<![\w.])(h\d+_\d+(?:_\d+)?)\b', p + r'\1', text)
    text = re.sub(r'(?<![\w.])(e_\d+_(?:\d+|concat_\d+))\b', p + r'\1', text)
    return re.sub(r"(?<![\w.])params(?![\w'])", "params'", text)
