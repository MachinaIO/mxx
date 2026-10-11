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
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if ch == ',' and depth == 0:
            out.append(cur.strip())
            cur = ''
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def stage_args(d, stage):
    claim = open(f'{d}/generated/Claim.lean').read()
    m = re.search(rf'  {re.escape(stage)}\.generatedRoot (?:Backend\.backend hashModel )?\w+ \(\((.*)\)\) execution\.«(stage_\d+)»', claim)
    args = _split_top(m.group(1))
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
    ins = []
    for line in body:
        m = re.match(r'let (\S+) : .*? := (inputs(?:\.2)*\.1)$', line)
        if m:
            ins.append((m.group(1), (len(m.group(2)) - len('inputs.1')) // 2))
    ins.sort(key=lambda x: x[1])
    return [w for w, _ in ins]


def root_outputs(path):
    _, names, atoms = parse(path, 'generatedRoot')
    out = dict(zip(names, atoms))['hout']
    return re.findall(r'\bw_\d+_\d+\b', out)
