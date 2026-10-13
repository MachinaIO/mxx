"""The wiring of the generated claim: for every stage, the source of each root input (a stage
output index or an external input), and the root `let` name of each input and output.

`stage_args(d, stage)` returns the list of argument sources `('stage', k, idx)` / `('ext', n)` of
the stage's root; `root_inputs(path)` the root input wires in order; `root_outputs(path)` the
output wires in order.
"""
import re

from patterns import _block, parse


def _split_top(s):
    out, depth, cur = [], 0, ''
    for ch in s:
        if ch in '([⟨':
            depth += 1
        elif ch in ')]⟩':
            depth -= 1
        if ch == ',' and depth == 0:
            out.append(cur.strip())
            cur = ''
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def merged_body_sources(d, stage):
    """For the view `Stage_<phase>_I` of a merged stage: the source of every input of the body
    (`stage_args` of the merged root, through its `∀ i : Fin 8, body i (…)` call): a family read at
    the instance is its root input's source, a packed family `('pack', sources of its members)`."""
    real = stage[:-len('_I')]
    path = f'{d}/generated/{real}.lean'
    args, num = stage_args(d, real)
    src = {w: args[i] for i, w in enumerate(root_inputs(path))}
    _, names, atoms = parse(path, 'generatedRoot')
    packs, call = {}, None
    for atom in atoms:
        m = re.match(r'MxxRuntime\.familyPack \(\d+\) \[([^\]]*)\] (w_\d+_\d+)$', atom)
        if m:
            packs[m.group(2)] = [x.strip() for x in m.group(1).split(',')]
        elif atom.startswith('(∀ i : Fin 8, parallel_generatedRoot_'):
            call = atom
    inner = call[call.index(' params i (') + len(' params i ('):]
    items = _leaves('(' + inner[:_close_args(inner)] + ')')
    out = []
    for item in items:
        if item == '()':
            continue
        w = re.search(r'\bw_\d+_\d+\b', item).group(0)
        out.append(('pack', tuple(src[x] for x in packs[w])) if w in packs else src[w])
    return out, num


def _close_args(s):
    depth = 1
    for k, ch in enumerate(s):
        depth += (ch == '(') - (ch == ')')
        if depth == 0:
            return k
    raise ValueError('unbalanced')


def _leaves(text):
    """The leaves of a tuple expression (nested pairs, `()` ends), in order."""
    text = text.strip()
    if text == '()':
        return []
    if text.startswith('(') and _close_args(text[1:]) == len(text) - 2:
        parts = _split_top(text[1:-1])
        if len(parts) > 1:
            return [leaf for part in parts for leaf in _leaves(part)]
    return [text]


def stage_args(d, stage):
    if stage.endswith('_I'):
        return merged_body_sources(d, stage)
    claim = open(f'{d}/generated/Claim.lean').read()
    m = re.search(rf'  {re.escape(stage)}\.generatedRoot (?:Backend\.backend hashModel )?\w+ \(\((.*)\)\) execution\.«(stage_\d+)»', claim)
    args = _leaves('(' + m.group(1) + ')')
    out = []
    for a in args:
        if a == '()':
            continue
        e = re.fullmatch(r'external\.input_(\d+)', a)
        s = re.fullmatch(r'execution\.«stage_(\d+)»((?:\.2)*)(\.1)?', a)
        if e:
            out.append(('ext', int(e.group(1))))
        elif s:
            idx = len(s.group(2)) // 2 if s.group(3) else None
            out.append(('stage', int(s.group(1)), idx))
        else:
            out.append(('?', a))
    return out, m.group(2)


def root_inputs(path):
    lines = open(path).read().split('\n')
    body = _block(lines, 'abbrev generatedRoot.body')
    # the inputs' lets come in input order (their projections may be balanced-tuple paths)
    ins = []
    for line in body:
        m = re.match(r'let (\S+) : .*? := (inputs(?:\.[12])*)$', line)
        if m:
            ins.append(m.group(1))
    return ins


def root_outputs(path):
    _, names, atoms = parse(path, 'generatedRoot')
    out = dict(zip(names, atoms))['hout']
    return re.findall(r'\bw_\d+_\d+\b', out)
