"""Destructuring patterns of generated Lean definitions.

A generated definition is a chain of `let`s and existentials whose body is a chain of
`constraints_k` abbreviations, each a conjunction of atoms. The functions here name the existential
witnesses as in the source (`w_<node>_<port>`), the meaningful atoms `h<node>_<port>` after the last
witness they mention, and the trivial bound atoms `-`, and turn those names into `rcases` patterns,
projections, or step-by-step destructuring tactics.
"""
import re

TRIVIAL = re.compile(
    r'^\(?-?\d+\)? (≤|<) \(?-?\d+\)?$|^\(\d+\) = \d+$|^0 ≤ \(\(Int.ofNat i_0\)\)$|'
    r'^\(\(Int.ofNat i_0\)\) < \d+$|^0 ≤ \(?w_\d+_0\)?$|^\(?w_\d+_0\)? < \d+$|^0 ≤ \(\d+\)$|'
    r'^\(\d+\) < \d+$')


def _body(lines, name):
    start = None
    for i, line in enumerate(lines):
        if line.startswith('abbrev ' + name + '.body ') or (line.startswith('def ' + name + ' ') and start is None):
            start = i
    body, j = [], start + 1
    while j < len(lines) and lines[j].startswith('  '):
        body.append(lines[j].strip())
        j += 1
    return body


def _atoms(lines, first):
    """The atoms of the constraint chain starting at `first`."""
    atoms, cur = [], first
    while cur:
        i = next(i for i, line in enumerate(lines) if line.startswith('abbrev ' + cur + ' '))
        nxt, k = None, i + 1
        while k < len(lines) and lines[k].startswith('  '):
            atom = lines[k].strip().rstrip('∧').strip()
            m = re.match(r'(\S+\.constraints_\d+) ', atom)
            if m:
                nxt = m.group(1)
            else:
                atoms.append(atom)
            k += 1
        cur = nxt
    return atoms


def _parse(path, name):
    lines = open(path).read().split('\n')
    body = _body(lines, name)
    witnesses, first = [], None
    for line in body:
        m = re.match(r'∃ \((w_\S+) : ', line)
        if m:
            witnesses.append(m.group(1))
        elif '.constraints_' in line and first is None:
            first = re.search(r'(\S+\.constraints_\d+)', line).group(1)
    atoms = _atoms(lines, first)
    used, names = {}, []
    for atom in atoms:
        if atom.startswith('outputs ='):
            names.append('hout')
            continue
        ws = re.findall(r'\bw_\d+_\d+(?:_concat_\d+)?\b', atom)
        if TRIVIAL.match(atom) or not ws:
            names.append('-')
            continue
        base = 'h' + ws[-1][2:]
        if base in used:
            used[base] += 1
            base = f'{base}_{used[base]}'
        else:
            used[base] = 0
        names.append(base)
    return witnesses, names, atoms


def pattern(path, name):
    """The `rcases` pattern of definition `name`: its witnesses, then its named atoms."""
    witnesses, names, _ = _parse(path, name)
    return '⟨' + ', '.join(witnesses + names) + '⟩'


def atom_kinds(path, name):
    """The head of the constraint of every named atom of definition `name`."""
    _, names, atoms = _parse(path, name)
    return {n: a.split(' ')[0] for n, a in zip(names, atoms) if n != '-'}


def _path(names, wanted):
    i = names.index(wanted)
    return '.2' * i + ('.1' if i < len(names) - 1 else '')


def proj(path, name, wanted, src='hb'):
    """`have` lines projecting the atoms `wanted` out of the destructured body `src`."""
    _, names, _ = _parse(path, name)
    return ''.join(f'  have {w} := {src}{_path(names, w)}\n' for w in wanted)


def destruct(path, name, hyp, keep):
    """Tactic lines that peel the existentials of definition `name` off `hyp` one at a time, then
    project the atoms `keep`."""
    witnesses, names, _ = _parse(path, name)
    out = []
    for w in witnesses:
        out.append(f'  try extract_lets at {hyp}')
        out.append(f'  obtain ⟨{w}, {hyp}⟩ := {hyp}')
    out.append(f'  try extract_lets at {hyp}')
    for w in keep:
        out.append(f'  have {w} := {hyp}{_path(names, w)}')
    return '\n'.join(out) + '\n'
