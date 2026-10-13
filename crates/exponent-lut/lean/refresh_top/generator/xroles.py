#!/usr/bin/env python3
"""Assign an L1 role (`GenIface.lean`) to every scope of an export and extract its parameters.

Roles are recognized from the scope's structure only: its atom multiset (`MxxRuntime.*`
predicates), the `let`-bound operations, the loop kinds and children, the tuple arities of its
inputs and outputs and how its parent calls it (root, par, gather, seq). Scope names, wire ids,
module-specific constants and counts are never used to decide a role (counts are extracted as
parameters). An unrecognized scope is an error.

Output (`--json out.json`): {module: {scope: {role, how, parent, count, params}}}.
Usage: xroles.py <generated dir> [--json out.json] [Module…]   (default: all proof modules)
"""
import collections
import json
import re
import sys

sys.path.insert(0, __file__.rsplit('/', 1)[0])
import xmodel  # noqa: E402

MODULES = ['Stage_preprocess', 'Stage_encrypt', 'Stage_online_eval', 'Stage_decode', 'Requirement_0']


def lets(s):
    c = collections.Counter()
    for w in s.wires.values():
        if w.src == 'let' and w.term:
            t = xmodel.strip_parens(w.term)
            h, _ = xmodel.app_split(t)
            h = h.split('.')[-1]
            if re.fullmatch(r'-?\d+', h):
                h = 'const'
            elif h.startswith('w_'):
                h = 'binop' if re.search(r' [-+*/%] ', t) else 'copy'
            c[h] += 1
    return c


def fam_len(t):
    m = re.match(r'^\(?Fin (\d+) → ', t or '')
    return int(m.group(1)) if m else None


def mat_dims(t):
    m = re.search(r'ExactMatrix (\S+|\([^()]*(?:\([^()]*\)[^()]*)*\)) (\d+) (\d+) (\d+)\s*\)?$', t or '')
    return (m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))) if m else None


class Ctx:
    def __init__(self, scopes):
        self.sc = scopes
        self.parent = {}
        for s in scopes.values():
            for a in s.loops:
                self.parent[a.loop.child] = (s.name, a.loop)


def role_of(cx, s):
    """(role, params) or (None, reason)."""
    sc = cx.sc
    par = cx.parent.get(s.name)
    how = par[1].kind if par else 'root'
    loop = par[1] if par else None
    A = s.atom_kinds()
    keys = set(A)
    L = lets(s)
    kids = [(a.loop.kind, sc.get(a.loop.child), a.loop) for a in s.loops]
    kin, kout = len(s.inputs), len(s.output)
    out_bool = s.output_type == 'Bool'
    out_int = s.output_type == 'Int'
    p = dict(how=how)
    if loop is not None:
        p['count'] = loop.count
    # ---- roots
    if how == 'root':
        if out_bool:
            return 'req.root', p
        if out_int:
            return 'decode.root', p
        return 'stage.root', p
    if how == 'par' and cx.parent[s.name][0] in sc and cx.parent.get(cx.parent[s.name][0]) is None \
            and s.binders == ['i_0'] and s.kind == 'record':
        return 'instance.body', p
    # ---- requirement (Bool state)
    if out_bool:
        if how != 'seq':
            return None, 'Bool scope outside a sequential loop'
        if kids:
            return 'req.nest.step', p
        return 'req.leaf.step', p
    # ---- integer sums (requirement inner sums, decode rounding)
    if out_int and how == 'seq' and not kids:
        if keys == {'familyGetDynamic'} and L.get('binop', 0) >= 1:
            if any('/' in (w.term or '') for w in s.wires.values()):
                return 'decode.round.step', p
            return 'req.int-sum.step', p
        return None, 'integer step of unknown shape'
    # the value table of a LUT key table: one scalar `selectInt` of the loop index (no inputs)
    if out_int and how == 'par' and not keys and not kids and kin == 0 and 'selectInt' in L:
        return 'lut.key-table.index', p
    if out_int and how == 'seq' and keys == {'sliceMatrix', 'polynomialValues'} and len(kids) == 1:
        return 'decode.column.step', p
    if s.module.startswith('Requirement'):
        return None, 'requirement scope of unknown shape'
    # ---- structural
    fams_in = [t for t in s.inputs if fam_len(t)]
    if how in ('par', 'gather') and keys == {'familyGetDynamic'} and not kids and \
            A['familyGetDynamic'] == kout and kin == kout and len(fams_in) == kin:
        p.update(k=kout, F=fam_len(s.inputs[0]))
        return 'gather.unpack', p
    if how == 'seq' and keys == {'familyGetDynamic'} and not kids and L.get('matrixAdd', 0) == A['familyGetDynamic']:
        k = A['familyGetDynamic']
        if kout == 0 or (kout == 1 and k == 1 and not s.output_type.startswith('(') and ' × ' not in s.output_type):
            p.update(k=1, F=fam_len(s.inputs[-1]), plain=True)
            return 'sum.members.step1', p
        if kout == k and kin == 2 * k:
            p.update(k=k, F=fam_len(s.inputs[k]))
            return 'sum.members.step', p
    if how == 'par' and keys == {'familyGetStatic'} and len(kids) == 2 and \
            [h for h, _, _ in kids] == ['par', 'seq'] and A['familyGetStatic'] == 2 * kout:
        ru = [role_of(cx, c)[0] for _, c, _ in kids]
        if ru == ['gather.unpack', 'sum.members.step']:
            p.update(k=kout, W=kids[0][2].count, F=fam_len(s.inputs[0]))
            return 'prf.rowsum', p
    if how == 'seq' and keys == {'familyGetDynamic', 'tensorRuns'} and len(kids) == 1 and \
            A['familyGetDynamic'] == 2 * kout and A['tensorRuns'] == kout:
        if role_of(cx, kids[0][1])[0] == 'sum.members.step':
            p.update(k=kout, W=kids[0][2].count + 2 if isinstance(kids[0][2].count, int) else None,
                     F=fam_len(s.inputs[kout]))
            return 'sum.columns.step', p
    if how == 'seq' and keys == {'sliceMatrix', 'tensorRuns'} and not kids and L.get('unitRow') == 1:
        return 'sum.columns.slice-step', p
    if how == 'par' and not keys and not kids and set(L) <= {'multiplyMonomial', 'fmod'} and \
            L.get('multiplyMonomial') == kout == kin:
        fm = [w.term for w in s.wires.values() if w.term and 'Int.fmod' in w.term]
        mm = re.search(r'Int\.fmod .* \((\d+)\)\)?$', fm[0]) if fm else None
        p.update(k=kout, N=int(mm.group(1)) if mm else None)
        return 'sampler.placement', p
    if how == 'par' and not keys and not kids and kin == 0 and set(L) <= {'selectInt', 'liftInteger', 'multiplyMonomial'}:
        return 'label.payload', p
    if how == 'par' and keys == {'hashSample'} and A['hashSample'] == 1 and not kids:
        return 'label.public', p
    # ---- encoding level (B owners)
    lutkeys = {'ringAutomorphismRuns', 'gadgetDecomposeRuns', 'hashSample', 'familyGetDynamic'}
    if how == 'seq' and lutkeys <= keys and not kids:
        p.update(k=kout)
        return ('lut.round.step' if kout > 4 else 'lut.key-round.step'), p
    if how in ('par', 'gather') and {'gadgetDecomposeRuns', 'hashSample', 'blockModSwitchRuns'} <= keys and \
            'ringAutomorphismRuns' not in keys and not kids:
        bucket1 = A.get('gadgetMatrixRuns', 0) >= 2 or A.get('centeredRebaseRuns', 0) >= 2
        key = 'gaussianSample' in keys
        p.update(k=kout)
        return ('prf.%scell%s' % ('key-' if key else '', '1' if bucket1 else '0')), p
    if how == 'par' and len(kids) == 1 and kids[0][0] == 'seq':
        cr = role_of(cx, kids[0][1])[0]
        if cr in ('lut.round.step', 'lut.key-round.step'):
            side = '' if cr == 'lut.round.step' else 'key-'
            # the smudge member weight `2^-1 P^digit` is the one power-of-base lift
            if any(w.src == 'let' and w.term and 'liftInteger' in w.term and ') ^ (' in w.term
                   for w in s.wires.values()):
                return f'smudge.{side}member', p
            if side and not keys:
                return ('lut.key-member.sampler' if L.get('matrixAdd') else 'lut.key-member.carrier'), p
            if side:
                return 'lut.key-member.readout', p
            if A.get('blockModSwitchRuns', 0) > 4:
                return 'lut.member.readout', p
            if L.get('matrixAdd', 0) > 8:
                return 'lut.member.sampler-offset', p
            return 'lut.member.sampler', p
    # a LUT key table's term step: sigma(X^(-j stride)) times the table value read at j
    if how == 'seq' and keys == {'familyGetDynamic', 'ringAutomorphismRuns'} and not kids and \
            kout == 1 and 'multiplyMonomial' in L and L.get('matrixAdd') == 1:
        p.update(representation='coefficient' if 'liftInteger' in L else 'exponent')
        return 'lut.key-table.step', p
    if how == 'par' and [h for h, _, _ in kids] == ['par', 'seq'] and 'gaussianSample' in keys and \
            kin == 3 and kout == 4:
        ru = [role_of(cx, c)[0] for _, c, _ in kids]
        if ru == ['lut.key-table.index', 'lut.key-table.step']:
            p.update(variant='centered' if 'centeredRebaseRuns' in keys else 'plain',
                     W=kids[1][2].count, representation=role_of(cx, kids[1][1])[1]['representation'])
            return 'lut.key-table', p
    if how == 'par' and not kids and 'gaussianSample' in keys and 'ringAutomorphismRuns' in keys and kout == 2:
        p.update(variant='reduce' if 'modulusReduceRuns' in keys else 'plain')
        return 'enc.lut-branch-key', p
    if how == 'par' and not kids and 'gaussianSample' in keys and 'liftInteger' in L:
        return 'enc.label-entry', p
    return None, f'unknown: how={how} kind={s.kind} in={kin} out={kout} atoms={dict(A)} lets={dict(L)} kids={[h for h, _, _ in kids]}'


def assign(gdir, modules):
    res, unknown = {}, []
    for m in modules:
        sc = xmodel.load(gdir, m)
        cx = Ctx(sc)
        res[m] = {}
        for name, s in sc.items():
            r, p = role_of(cx, s)
            if r is None:
                unknown.append((m, name, p))
                continue
            par = cx.parent.get(name)
            res[m][name] = dict(role=r, how=p.pop('how'), parent=par[0] if par else None,
                                count=p.pop('count', None), params=p)
    return res, unknown


if __name__ == '__main__':
    args = sys.argv[1:]
    gdir = args.pop(0)
    out = None
    if '--json' in args:
        i = args.index('--json')
        out = args[i + 1]
        del args[i:i + 2]
    res, unknown = assign(gdir, args or MODULES)
    for m, d in res.items():
        c = collections.Counter(v['role'] for v in d.values())
        print(f'{m}: {len(d)} scopes, roles ' + ', '.join(f'{r}:{n}' for r, n in sorted(c.items())))
    if out:
        json.dump(res, open(out, 'w'), indent=1)
    for m, n, why in unknown:
        print(f'UNKNOWN {m} {n}: {why}')
    print(f'{len(unknown)} unknown')
    sys.exit(1 if unknown else 0)
