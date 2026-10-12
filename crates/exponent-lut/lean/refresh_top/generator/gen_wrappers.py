"""Write `StageWrappers.lean`: every merged stage's root relation gives its body at each instance.

A merged stage root packs the per-instance protocol inputs of its parts (`familyPack`), runs
`∀ i : Fin 8, parallel_generatedRoot_K … i (args i) (outs i)` and returns the loop's output
families (and the unpacked members a later stage reads). `<stage>_at` states the body at instance
`i` with its arguments and outputs read from the root's inputs and outputs.

Usage: `python3 gen_wrappers.py <parameter directory>`.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from patterns import _block, parse

WIRE = re.compile(r'\bw_\d+_\d+\b')


def tuple_paths(text, prefix):
    """Projection paths of the leaves of a nested pair expression `text` (no `()` leaves)."""
    text = text.strip()
    if not text.startswith('('):
        return {text: prefix}
    inner = text[1:-1]
    depth, split = 0, None
    for k, ch in enumerate(inner):
        if ch in '([':
            depth += 1
        elif ch in ')]':
            depth -= 1
        elif ch == ',' and depth == 0:
            split = k
            break
    if split is None:
        return tuple_paths(inner, prefix)
    left, right = inner[:split], inner[split + 1:].strip()
    out = tuple_paths(left, prefix + '.1')
    if right != '()':
        if right.startswith('(') and ',' not in right[1:-1] and right != '()':
            out.update(tuple_paths(right, prefix + '.2'))
        else:
            out.update(tuple_paths(right, prefix + '.2'))
    return out


def stage(d, name):
    path = f'{d}/generated/Stage_{name}.lean'
    ns = f'Stage_{name}'
    text = open(path).read()
    lines = text.split('\n')
    body = _block(lines, 'abbrev generatedRoot.body')
    inputs = {}
    for line in body:
        m = re.match(r'let (w_\d+_\d+) : .* := (inputs[.12]*)$', line)
        if m:
            inputs[m.group(1)] = m.group(2)
    _, names, atoms = parse(path, 'generatedRoot')
    packs, call, outs = {}, None, None
    for atom in atoms:
        m = re.match(r'MxxRuntime\.familyPack \(\d+\) \[([^\]]*)\] (w_\d+_\d+)$', atom)
        if m:
            packs[m.group(2)] = [x.strip() for x in m.group(1).split(',')]
        elif atom.startswith('(∀ i : Fin 8, parallel_generatedRoot_'):
            call = atom
        elif atom.startswith('outputs ='):
            outs = atom[len('outputs ='):].strip()
    k = re.match(r'\(∀ i : Fin 8, parallel_generatedRoot_(\d+) ', call).group(1)
    out_paths = tuple_paths(outs, 'outputs')
    inner = call[len('(∀ i : Fin 8, '):-1]
    inner = inner.replace(f'parallel_generatedRoot_{k} backend hashModel params i ',
                          f'{ns}.parallel_generatedRoot_{k} Backend.backend model params i ', 1)

    def member(w):
        if w in inputs:
            return inputs[w]
        raise KeyError(w)

    def applied(m):
        w = m.group(1)
        if w in inputs:
            return f'({inputs[w]} i)'
        if w in packs:
            return f"([{', '.join(member(x) for x in packs[w])}].get ⟨i.val, i.isLt⟩)"
        if w in out_paths:
            return f'({out_paths[w]} i)'
        raise KeyError(w)

    inner = re.sub(r'\((w_\d+_\d+) i\)', applied, inner)
    inner = WIRE.sub(lambda m: inputs[m.group(0)], inner)
    pack_names = [n for n, a in zip(names, atoms) if a.startswith('MxxRuntime.familyPack')]
    pattern = []
    for n, a in zip(names, atoms):
        if a.startswith('MxxRuntime.familyPack'):
            pattern.append(n)
        elif a.startswith('(∀ i : Fin 8'):
            pattern.append('hall')
        elif a.startswith('outputs ='):
            pattern.append('hout')
        else:
            pattern.append('-')
    constraints = sorted(set(re.findall(r'abbrev (generatedRoot\.constraints_\d+) ', text)),
                         key=lambda c: int(c.rsplit('_', 1)[1]))
    unfold = ', '.join(f'{ns}.{c}' for c in constraints)
    rewrites = ', '.join(f'familyPack_get {n}' for n in pack_names)
    return f'''set_option maxHeartbeats 0 in
set_option maxRecDepth 65536 in
/-- The {name} stage's body at instance `i`. -/
theorem {name}_at {{model : HashModel}} {{params : {ns}.Params}} {{inputs outputs}}
    (h : {ns}.generatedRoot Backend.backend model params inputs outputs) (i : Fin 8) :
    {inner} := by
  obtain ⟨w, h⟩ := h
  unfold {ns}.generatedRoot.body at h
  simp only [{unfold}] at h
  obtain ⟨{', '.join(pattern)}⟩ := h
  subst hout
  have h := hall i
  {f'simp only [{rewrites}] at h' if rewrites else 'skip'}
  exact h
'''


if __name__ == '__main__':
    d = sys.argv[1]
    which = sys.argv[2:] or ['encrypt', 'preprocess', 'online_eval']
    out = ['import TopStage\n']
    out += [f'import Stage_{s}\n' for s in which]
    out.append('''
/-! The merged stages at an instance (written by `generator/gen_wrappers.py`). -/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime

''')
    for s in which:
        out.append(stage(d, s) + '\n')
    out.append('end RefreshTopProof\n')
    open(f'{d}/StageWrappers.lean', 'w').write(''.join(out))
