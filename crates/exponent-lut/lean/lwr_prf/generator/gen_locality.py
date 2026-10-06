#!/usr/bin/env python3
"""Write the locality lemmas of a generated stage.

Usage: gen_locality.py <parameter dir> <stage> <head>

For every relation of `generated/<stage>.lean` that reads the sample tape, the module
`<stage>Locality.lean` states that the relation on the tape restricted to the stage's keys (those
whose site starts with `<head>`, the stage's position in the workflow) is the relation on the full
tape. The lemmas are proved bottom-up: a relation unfolds to its constraints, its tape samples
read keys of the restricted universe, and its children are covered by their own lemmas. A relation
that ignores the tape has the lemma `Iff.rfl`.
"""
import os
import re
import sys

DEF = re.compile(r'^(def|abbrev) (\S+) (.*) : Prop :=$')
BINDER = re.compile(r'\((\w+) : ')
SAMPLE = re.compile(r'MxxRuntime\.(gaussianSampleAt|uniformIntervalSampleAt) tape \(path \+\+ \[(\d+)\]\) '
                    r'(\(\([^)]*\)\)|\([^)]*\)) \(([^)]*)\) (\w+)')
CHILD = re.compile(r'((?:parallel|sequential)_generatedRoot_\d+) backend \w+ tape \(path \+\+ \[(\d+), [^\]]+\]\)')
MATRIX_BINDER = r'\(?{w} : (?:Fin \d+ → )?Mxx\.Primitives\.ExactMatrix \d+ (\d+) (\d+) (\d+)'


def parse(text):
    """The relations of a stage in file order: name, kind, binder names, and the text of their
    definition and constraints."""
    lines = text.split('\n')
    defs = []
    for i, line in enumerate(lines):
        m = DEF.match(line)
        if m:
            defs.append((m.group(1), m.group(2), m.group(3), i))
    blocks = {}
    for k, (kind, name, header, start) in enumerate(defs):
        end = defs[k + 1][3] if k + 1 < len(defs) else len(lines)
        blocks[name] = (kind, header, '\n'.join(lines[start:end]))
    return defs, blocks


def roots(defs):
    return [name for kind, name, _, _ in defs if kind == 'def']


def owned(name, blocks):
    """The definition of a relation and its constraint abbreviations."""
    return [n for n in blocks if n == name or n.startswith(name + '.')]


def law_of(kind, a, b):
    if kind == 'gaussianSampleAt':
        return f'.gaussian {a} {b}'
    return f'.interval {a} {b}'


def analyze(defs, blocks):
    info = {}
    for name in roots(defs):
        _, header, _ = blocks[name]
        reads = '(tape : MxxRuntime.SampleTape)' in header
        text = '\n'.join(blocks[n][2] for n in owned(name, blocks))
        samples = [(m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)) for m in SAMPLE.finditer(text)]
        children = sorted({(m.group(1), m.group(2)) for m in CHILD.finditer(text)})
        info[name] = dict(header=header, reads=reads, samples=samples, children=children, text=text)
    depth = {}
    for name in roots(defs):
        d = 1 if info[name]['samples'] else 0
        for child, _ in info[name]['children']:
            if info[child]['reads']:
                d = max(d, 2 + depth[child])
        depth[name] = d
    return info, depth


def binders(header):
    names = []
    for k, name in enumerate(BINDER.findall(header)):
        names.append(name if name != '_' else f'x_{k}')
    # `_` binders are positional: rename them in order of appearance.
    out, k = [], 0
    for m in re.finditer(r'\((\w+) : ', header):
        out.append(m.group(1) if m.group(1) != '_' else f'x_{k}')
        k += 1
    return out


def lemma(stage, name, info, depth, universe):
    i = info[name]
    args = binders(i['header'])
    tape_index = args.index('tape') if 'tape' in args else next(
        k for k, a in enumerate(args) if a.startswith('x_') and k == 2)
    restricted = [f'(restrictTape {universe} {a})' if k == tape_index else a for k, a in enumerate(args)]
    full = list(args)
    binder_text = ' '.join('{' + a + '}' for a in args)
    lhs = f'{stage}.{name} ' + ' '.join(restricted)
    rhs = f'{stage}.{name} ' + ' '.join(full)
    if not i['reads']:
        return (f'set_option maxRecDepth 100000 in\n'
                f'theorem {name}_restrict {binder_text} :\n    {lhs} ↔\n      {rhs} := by\n'
                f'  unfold {stage}.{name}\n  exact Iff.rfl\n')
    heads = [n for n in owned_names[name] if n == name or n.endswith('.body')]
    heads.sort(key=lambda n: n != name)
    rest = [n for n in owned_names[name] if n not in heads]
    unfold = ' '.join(f'{stage}.{n}' for n in heads)
    if rest:
        unfold += ('\n  repeat (first ' + ' '.join(f'| unfold {stage}.{n}' for n in rest) + ')')
    children = sorted({c for c, _ in i['children']})
    kinds = sorted({f'{k}_restrict' for k, _, _, _, _ in i['samples']})
    simps = ', '.join(kinds + [f'{c}_restrict' for c in children])
    return (f'set_option maxHeartbeats 16000000 in\n'
            f'set_option maxRecDepth 100000 in\n'
            f'theorem {name}_restrict {binder_text}\n'
            f'    (hp : GoodPath {HEADS} {L} {M} {depth[name]} path) :\n'
            f'    {lhs} ↔\n      {rhs} := by\n'
            f'  unfold {unfold}\n'
            f'  simp (config := {{zeta := false, maxSteps := 100000000}}) (disch := locality_disch) only [{simps}]\n')


def main():
    global HEADS, L, M, owned_names
    directory, stage, head = sys.argv[1], sys.argv[2], int(sys.argv[3])
    text = open(os.path.join(directory, 'generated', f'{stage}.lean')).read()
    defs, blocks = parse(text)
    info, depth = analyze(defs, blocks)
    owned_names = {name: [n for n in owned(name, blocks)] for name in roots(defs)}
    laws = sorted({law_of(k, a, b) for i in info.values() for k, _, a, b, _ in i['samples']})
    HEADS, L, M = f'[{head}]', 1 + max(depth.values()), 100000
    universe = f'{stage}Universe'
    out = [f'import {stage}\nimport ExponentFunctional\n',
           '/-!\n# Locality of `' + stage + '`\n\nGenerated by `generator/gen_locality.py`. The stage '
           'reads the tape only at keys of\n`' + universe + '`: its relations on the restricted tape '
           'are its relations on the full tape.\n-/\n',
           'namespace LwrProof\n\nopen MxxRuntime MxxExponentLut\n',
           f'/-- The keys that `{stage}` may read. -/\nabbrev {universe} : Finset SampleKey :=\n'
           f'  sampleUniverse {HEADS} {L} {M} 4096 4096 65536 [{", ".join(laws)}]\n',
           'macro "locality_disch" : tactic => `(tactic| first\n'
           '  | (refine GoodPath.siteKeysIn ‹GoodPath _ _ _ _ _› ?_ ?_ ?_ ?_ ?_ ?_ <;> first\n'
           '      | decide\n'
           '      | (intro x hx; simp only [List.mem_cons, List.not_mem_nil, or_false] at hx; omega)\n'
           '      | simp only [List.mem_cons, List.mem_singleton, true_or, or_true])\n'
           '  | (refine GoodPath.append ‹GoodPath _ _ _ _ _› ?_ ?_ <;> first\n'
           '      | (simp only [List.length_cons, List.length_nil]; omega)\n'
           '      | (intro x hx; simp only [List.mem_cons, List.not_mem_nil, or_false] at hx; omega)))\n']
    for name in roots(defs):
        out.append(lemma(stage, name, info, depth, universe))
    out.append('end LwrProof\n')
    path = os.path.join(directory, f'{stage}Locality.lean')
    with open(path, 'w') as f:
        f.write('\n'.join(out))
    print('wrote', path, 'root depth', depth['generatedRoot'])


if __name__ == '__main__':
    main()
