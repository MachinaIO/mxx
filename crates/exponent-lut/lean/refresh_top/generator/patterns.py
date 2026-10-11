"""Destructuring patterns of generated Lean definitions.

A generated definition `X` is `∃ witness, X.body ... witness`, whose body is a chain of `let`s
ending in a chain of `constraints_k` abbreviations, each a conjunction of atoms. `lets` names the
`let`s as in the source, for `extract_lets`; `atoms` names the meaningful atoms `h<node>_<port>`
after the last wire they mention and the trivial bound atoms `-`, for `obtain`.
"""
import re

TRIVIAL = re.compile(
    r'^\(?-?\d+\)? (≤|<) \(?-?\d+\)?$|^\(\d+\) = \d+$|^0 ≤ \(\(Int.ofNat i_0\)\)$|'
    r'^\(\(Int.ofNat i_0\)\) < \d+$|^0 ≤ \(\d+\)$|^\(\d+\) < \d+$')


def _block(lines, head):
    i = next(i for i, line in enumerate(lines) if line.startswith(head + ' '))
    out, j = [], i + 1
    while j < len(lines) and lines[j].startswith('  '):
        out.append(lines[j].strip())
        j += 1
    return out


def _atoms(lines, first):
    atoms, cur = [], first
    while cur:
        nxt = None
        for line in _block(lines, 'abbrev ' + cur):
            atom = line.rstrip('∧').strip()
            m = re.match(r'(\S+\.constraints_\d+) ', atom)
            if m:
                nxt = m.group(1)
            else:
                atoms.append(atom)
        cur = nxt
    return atoms


def parse(path, name):
    """The `let` names and the atom names of definition `name` (without its namespace) of file
    `path`."""
    lines = open(path).read().split('\n')
    body = _block(lines, 'abbrev ' + name + '.body')
    lets, first = [], None
    for line in body:
        m = re.match(r'let (\S+)', line)
        if m:
            lets.append(m.group(1))
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
    return lets, names, atoms


def destruct(path, namespace, name, hyp):
    """Tactics turning `hyp : namespace.name ...` into named lets and atoms."""
    lets, names, _ = parse(path, name)
    text = open(path).read()
    constraints = sorted(set(re.findall(r'abbrev (' + re.escape(name) + r'\.constraints_\d+) ', text)))
    unfold = ', '.join(f'{namespace}.{c}' for c in constraints)
    return (f'  obtain ⟨w, {hyp}⟩ := {hyp}\n'
            f'  unfold {namespace}.{name}.body at {hyp}\n'
            f'  extract_lets -merge {" ".join(lets)} at {hyp}\n'
            f'  simp only [{unfold}] at {hyp}\n'
            f'  obtain ⟨{", ".join(names)}⟩ := {hyp}\n')


def destruct_root(path, namespace, name, hyp):
    """`destruct` for a large definition: unfold the `constraints_k` chain one abbreviation at a
    time and name its atoms, so no tactic sees the whole conjunction."""
    lets, names, _ = parse(path, name)
    lines = open(path).read().split('\n')
    body = _block(lines, 'abbrev ' + name + '.body')
    cur = next(re.search(r'(\S+\.constraints_\d+)', l).group(1) for l in body if '.constraints_' in l)
    out = [f'  obtain ⟨w, {hyp}⟩ := {hyp}\n',
           f'  unfold {namespace}.{name}.body at {hyp}\n',
           f'  extract_lets -merge {" ".join(lets)} at {hyp}\n']
    k = 0
    while cur:
        nxt, count = None, 0
        for line in _block(lines, 'abbrev ' + cur):
            m = re.match(r'(\S+\.constraints_\d+) ', line.rstrip('∧').strip())
            if m:
                nxt = m.group(1)
            else:
                count += 1
        chunk = names[k:k + count]
        k += count
        out.append(f'  unfold {namespace}.{cur} at {hyp}\n')
        out.append(f'  obtain ⟨{", ".join(chunk + ([hyp] if nxt else []))}⟩ := {hyp}\n')
        cur = nxt
    assert k == len(names)
    return ''.join(out)


def _def_block(lines, name):
    """The body lines of `def name (...) : Prop :=`."""
    return _block(lines, 'def ' + name)


def parse_inline(path, name):
    """For a definition `def name ... := lets, ∃ witnesses, ..., constraints`: the destructuring
    steps (`('lets', [names])` and `('exists', name)` in order), the atom names and the atoms.

    `extract_lets` lifts every `let` whose value mentions no witness that is still bound, in
    order of appearance, so the steps name the lets in that order: after the initial extraction
    and after every `obtain`, the lets whose dependencies are all available."""
    lines = open(path).read().split('\n')
    body = _def_block(lines, name)
    items, first = [], None
    for line in body:
        m = re.match(r'let (\S+) :[^:]*?:= (.*)$', line) or re.match(r'let (\S+) := (.*)$', line)
        if m:
            items.append(('let', m.group(1), set(re.findall(r'\bw_\d+_\d+(?:_concat_\d+)?\b', m.group(2)))))
            continue
        m = re.match(r'∃ \((\S+) : ', line)
        if m:
            items.append(('exists', m.group(1), set()))
            continue
        if '.constraints_' in line and first is None:
            first = re.search(r'(\S+\.constraints_\d+)', line).group(1)
    witnesses = {x for k, x, _ in items if k == 'exists'}
    available, done, steps = set(), set(), []

    def liftable():
        out = []
        for k, x, deps in items:
            if k == 'let' and x not in done and all(d in available or d not in witnesses
                                                    and (d in done or d not in
                                                         {y for kk, y, _ in items if kk == 'let'})
                                                    for d in deps):
                # Lets depending on not-yet-extracted lets wait for them.
                out.append(x)
                done.add(x)
        return out

    # Lets may depend on earlier lets: iterate until stable.
    def lift_all():
        names = []
        while True:
            more = liftable()
            if not more:
                return names
            names += more

    first_lets = lift_all()
    if first_lets:
        steps.append(('lets', first_lets))
    for k, x, _ in items:
        if k == 'exists':
            available.add(x)
            steps.append(('exists', x))
            more = lift_all()
            if more:
                steps.append(('lets', more))
    atoms = _atoms(lines, first)
    used, names = {}, []
    for atom in atoms:
        if atom.startswith('outputs ='):
            names.append('hout')
            continue
        ws = re.findall(r'\bw_\d+_\d+(?:_concat_\d+)?\b', atom)
        if TRIVIAL.match(atom) or not ws or re.match(r'^\(\d+\) ≠ 0$', atom):
            names.append('-')
            continue
        base = 'h' + ws[-1][2:]
        if base in used:
            used[base] += 1
            base = f'{base}_{used[base]}'
        else:
            used[base] = 0
        names.append(base)
    return steps, names, atoms


def destruct_inline(path, namespace, name, hyp):
    """Tactics turning `hyp : namespace.name ...` (an inline-existential definition) into named
    lets, witnesses and atoms."""
    steps, names, _ = parse_inline(path, name)
    text = open(path).read()
    constraints = sorted(set(re.findall(r'abbrev (' + re.escape(name) + r'\.constraints_\d+) ', text)))
    unfold = ', '.join(f'{namespace}.{c}' for c in constraints)
    out = [f'  unfold {namespace}.{name} at {hyp}']
    for kind, x in steps:
        if kind == 'lets':
            out.append(f'  extract_lets -merge {" ".join(x)} at {hyp}')
        else:
            out.append(f'  obtain ⟨{x}, {hyp}⟩ := {hyp}')
    out.append(f'  simp only [{unfold}] at {hyp}')
    out.append(f'  obtain ⟨{", ".join(names)}⟩ := {hyp}')
    return '\n'.join(out) + '\n'
