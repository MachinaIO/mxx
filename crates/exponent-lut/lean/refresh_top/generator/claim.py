"""The wiring of the generated claim: the arguments of every stage relation in `Runs`.

`stage_args(d, k)` lists the argument expressions of stage `k` (`execution.«stage_k»`), each
one of `('external', i)`, `('stage', s, p)` (projection `p` of stage `s`'s outputs) or
`('unit',)`.
"""
import re


def _split(s):
    """Top-level comma-separated items of `s`."""
    out, depth, cur = [], 0, ''
    for ch in s:
        if ch in '([':
            depth += 1
        elif ch in ')]':
            depth -= 1
        if ch == ',' and depth == 0:
            out.append(cur.strip())
            cur = ''
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def stage_lines(d):
    text = open(f'{d}/generated/Claim.lean').read()
    runs = text[text.index('def Runs'):text.index('def CorrectnessClaim')]
    return [l.strip() for l in runs.split('\n') if '.generatedRoot ' in l]


def stage_args(d, k):
    line = stage_lines(d)[k]
    m = re.search(r'params \(\((.*)\)\) execution', line)
    if not m:
        m = re.search(r'params \((.*)\) execution', line)
    items = _split(m.group(1))
    out = []
    for it in items:
        if it == '()':
            out.append(('unit',))
        elif it.startswith('external.input_'):
            out.append(('external', int(it.split('_')[1])))
        else:
            mm = re.match(r'execution\.«stage_(\d+)»((?:\.\d)*)', it)
            if mm:
                p = mm.group(2).count('.2') if mm.group(2) else -1
                out.append(('stage', int(mm.group(1)), p, it))
            else:
                out.append(('other', it))
    return out


def stage_name(d, k):
    return re.match(r'(Stage_\S+)\.generatedRoot', stage_lines(d)[k]).group(1)


def decode_externals(d):
    """The external inputs of the decode stage by role, as `input_<i>` names: `sigma` (the
    coefficients of `σ₀`), `key` (the hash key of `A'`), `c` and `ep` (the expected message's
    coefficient and exponent), and the source input's exponent `e`, the one external before `c`."""
    k = next(k for k, l in enumerate(stage_lines(d)) if l.startswith('Stage_decode.'))
    sigma, key, c, ep = [f'input_{a[1]}' for a in stage_args(d, k) if a[0] == 'external']
    return {'sigma': sigma, 'key': key, 'c': c, 'ep': ep, 'e': f'input_{int(c[6:]) - 1}'}
