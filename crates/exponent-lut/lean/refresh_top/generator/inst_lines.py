"""The body relations of instance `i0` of the merged stages, as the instance's assembly states them.

`body_lines(d)` returns, for every merged phase, the relation
`Stage_<phase>.parallel_generatedRoot_K Backend.backend model GeneratedClaim.stage_<k>_params i0 ARGS y<k>`
(from `StageWrappers.lean`, `<phase>_at`) where `y<k>` is the instance's body output tuple, an input
read from another phase's family is that phase's body output at the instance (`y<m>.<path>`), a
broadcast input is its claim argument (`external.input_n`, `out<m>` for payload and mask) and a
packed input `[…].get ⟨i0, hi0⟩` of its claim arguments. Also returns the stage numbers.
"""
import re

from gen_wrappers import tuple_paths
from patterns import _block, parse
from wiring import _leaves

PHASES = ('preprocess', 'encrypt', 'online_eval')


def body_projection(name, position, total):
    """The exporter's projection of component `position` of a `total`-tuple (`tuple_projection`)."""
    if total == 1:
        return name
    if total > 256:
        half = total // 2
        if position < half:
            return body_projection(f'{name}.1', position, half)
        return body_projection(f'{name}.2', position - half, total - half)
    out = name + '.2' * position
    return out + ('.1' if position < total - 1 else '')


def claim_args(d, phase):
    claim = open(f'{d}/generated/Claim.lean').read()
    m = re.search(rf'  Stage_{phase}\.generatedRoot Backend\.backend hashModel (stage_(\d+)_params) \(\((.*)\)\) execution\.«stage_\d+»', claim)
    return int(m.group(2)), _leaves('(' + m.group(3) + ')')


def root_paths(d, phase):
    """The root input lets of a merged stage: wire to projection of `inputs`."""
    lines = open(f'{d}/generated/Stage_{phase}.lean').read().split('\n')
    body = _block(lines, 'abbrev generatedRoot.body')
    out = []
    for line in body:
        m = re.match(r'let (w_\d+_\d+) : .* := (inputs(?:\.[12])*)$', line)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def body_lines(d):
    wrappers = open(f'{d}/StageWrappers.lean').read()
    nums = {p: claim_args(d, p)[0] for p in PHASES}
    # each phase's root outputs: projection path -> body output port and the body's output count
    out_ports, out_total = {}, {}
    for p in PHASES:
        _, names, atoms = parse(f'{d}/generated/Stage_{p}.lean', 'generatedRoot')
        outs = dict(zip(names, atoms))['hout'][len('outputs ='):].strip()
        paths = tuple_paths(outs, '')
        k = re.search(r'∀ i : Fin \d+, parallel_generatedRoot_(\d+) ', open(f'{d}/generated/Stage_{p}.lean').read()).group(1)
        ports = {}
        for w, path in paths.items():
            m = re.fullmatch(rf'w_{k}_(\d+)', w)
            if m:
                ports[path] = int(m.group(1))
        out_ports[nums[p]] = ports
        # the tuple ends with `()`
        out_total[nums[p]] = max(ports.values()) + 2
    lines = {}
    for p in PHASES:
        num, args = claim_args(d, p)
        leaf = {path[len('inputs'):]: args[i] for i, (_, path) in enumerate(root_paths(d, p))}
        m = re.search(rf'theorem {p}_at .*?:\n    (.*?) := by', wrappers, re.S)
        text = ' '.join(m.group(1).split())

        def source(path, applied):
            a = leaf[path]
            s = re.fullmatch(r'execution\.«stage_(\d+)»((?:\.[12])*)', a)
            if s and int(s.group(1)) in out_ports and applied:
                port = out_ports[int(s.group(1))][s.group(2)]
                return body_projection(f'y{s.group(1)}', port, out_total[int(s.group(1))])
            if s:
                return f'out{s.group(1)}{s.group(2)}'
            return a
        text = re.sub(r'\(inputs((?:\.[12])*) i\)', lambda m: source(m.group(1), True), text)
        text = re.sub(r'\binputs((?:\.[12])*)', lambda m: source(m.group(1), False), text)
        text = text.replace('⟨i.val, i.isLt⟩', '⟨i0, hi0⟩')
        text = text.replace(' Backend.backend model params i ', f' Backend.backend model GeneratedClaim.stage_{num}_params i0 ')
        # the outputs: the instance's output tuple
        head, _, rest = text.partition(f'GeneratedClaim.stage_{num}_params i0 ')
        args_text = rest[:_close(rest) + 1]
        lines[p] = f'{head}GeneratedClaim.stage_{num}_params i0 {args_text} y{num}'
    return lines, nums


def _close(s):
    depth = 0
    for k, ch in enumerate(s):
        depth += (ch == '(') - (ch == ')')
        if depth == 0:
            return k
    raise ValueError('unbalanced')
