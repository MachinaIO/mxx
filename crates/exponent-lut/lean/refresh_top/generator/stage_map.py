"""The source of every root input of a stage of part `c_j`: a preprocess or encrypt output wire,
an external input, or another stage, from `generated/Claim.lean`."""
import re

from chain import WIRE
from patterns import parse

STAGES = {'payload': 0, 'mask': 1}


def split_top(s):
    """The top-level comma-separated items of a tuple body."""
    out, depth, cur = [], 0, ''
    for ch in s:
        if ch in '([{⟨':
            depth += 1
        elif ch in ')]}⟩':
            depth -= 1
        if ch == ',' and depth == 0:
            out.append(cur.strip())
            cur = ''
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def outputs_of(d, stage, part):
    """The output wires of a stage root, in tuple order."""
    path = f'{d}/generated/Stage_{stage}_{part}.lean'
    _, names, atoms = parse(path, 'generatedRoot')
    hout = dict(zip(names, atoms))['hout']
    return WIRE.findall(hout)


def stage_numbers(d):
    claim = open(f'{d}/generated/Claim.lean').read()
    return {m.group(2): int(m.group(1)) for m in
            re.finditer(r'def stage_(\d+)_params : (\S+)\.Params', claim)}


def merged_input_sources(d, stage):
    """`input_sources` of the body of a merged stage (its view `Stage_<stage>_I`): a family read
    at the instance comes from the producer body's output of the same port."""
    from wiring import merged_body_sources, root_inputs, _leaves
    nums = {v: k for k, v in stage_numbers(d).items()}
    args, _ = merged_body_sources(d, f'Stage_{stage}_I')
    ins = root_inputs(f'{d}/generated/Stage_{stage}_I.lean')
    out = []
    for w, a in zip(ins, args):
        if a[0] == 'stage' and a[2] is not None:
            name = nums[a[1]]
            _, names, atoms = parse(f'{d}/generated/{name}.lean', 'generatedRoot')
            outs = _leaves(dict(zip(names, atoms))['hout'][len('outputs ='):])
            port = int(outs[a[2]].rsplit('_', 1)[1])
            out.append((w, ('stage', name, port)))
        else:
            out.append((w, ('external', a)))
    return out


def input_sources(d, stage, part):
    """`[(input wire, source)]` with source `('stage', name, index)` or `('external', k)`."""
    if part == 'I':
        return merged_input_sources(d, stage)
    claim = open(f'{d}/generated/Claim.lean').read()
    nums = {v: k for k, v in stage_numbers(d).items()}
    ns = f'Stage_{stage}_{part}'
    line = next(l for l in claim.split('\n') if l.strip().startswith(f'{ns}.generatedRoot'))
    i = line.index('((') + 2
    j = line.rindex(')) execution')
    items = split_top(line[i:j])
    items = [x for x in items if x != '()']
    path = f'{d}/generated/{ns}.lean'
    text = open(path).read()
    body = text[text.index('abbrev generatedRoot.body'):]
    ins = [w for w, _ in re.findall(r'let (w_\d+_0) : ([^\n]*?) := inputs', body)]
    out = []
    for w, it in zip(ins, items):
        m = re.match(r'execution\.«stage_(\d+)»((?:\.2)*)\.1$', it) or re.match(r'execution\.«stage_(\d+)»((?:\.2)*)$', it)
        if m:
            out.append((w, ('stage', nums[int(m.group(1))], len(m.group(2)) // 2)))
        else:
            out.append((w, ('external', it)))
    return out
