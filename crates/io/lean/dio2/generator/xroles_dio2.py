#!/usr/bin/env python3
"""The L1 roles of refresh_top (`xroles.py`) extended by the scope shapes only Diamond iO 2 has.

Importing this module replaces `xroles.role_of`, so refresh_top's generators (`gen_adapters.py`)
classify the dio2 stages too. The new roles, recognized from structure only as in `xroles.py`:

| Role | Shape |
| --- | --- |
| `lut.member.flat` | a root parallel over the members of one LUT round at one modulus |
| `lut.key-member.flat` | the same in a preprocessing |
| `lut.key-table.flat` | a root parallel over a LUT key table encrypted without error |
| `import.copy` | a parallel body that forwards its inputs (an imported row's two ports) |
| `import.label` | a parallel body that forwards two rows and lifts `c X^e` (an imported label) |
| `sampler.monomial` | a parallel body of monomials `X^e` of its integer inputs (Sign/CBD placement) |
| `prf.cell-flat` | a PRF cell at one modulus: a label member times its state, times the selector (no move down) |
| `preimage.sample` | one preimage per family member |
| `enc.rooted-branch-key` | the key of a rooted LUT branch (preprocess) |
| `enc.rooted-label-entry` | the key of a rooted label entry (preprocess) |

V2 also recognizes symbolic `label.forest.*` scopes, `label.zero`, scalar gathers,
fused gather/monomial placement, and block-diagonal smudge digit steps and weights.
Port aliases and matrix/numeral notations are resolved across imported partitions.
Definitions without a loop parent are reported explicitly as `unused.body`.

Usage: xroles_dio2.py <generated dir> [--json out.json] [Module…]   (default: every stage)
"""
import os
import re
import sys

REFRESH_TOP = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '../../../../exponent-lut/lean/refresh_top/generator')
sys.path.insert(0, REFRESH_TOP)
import xroles  # noqa: E402
import xmodel  # noqa: E402

_sources = xmodel.module_sources


def export_sources(gdir, module):
    """Resolve port aliases across all partitions before parsing their signatures."""
    sources = _sources(gdir, module)
    aliases = dict(match for _, text in sources for match in re.findall(
        r'^abbrev (\w+\.(?:InputType|OutputType)) : Type :=\n  ([^\n]+)', text, re.M))
    notations = '\n'.join(line for _, text in sources for line in re.findall(
        r'^local notation "mxx(?:Nat|Mat)\d+" => [^\n]+$', text, re.M))
    prefix = notations + '\n' if len(sources) > 1 else ''
    return [(name, prefix + re.sub(r'\((\w+\.(?:InputType|OutputType))\)',
                         lambda match: '(' + aliases.get(match[1], match[1]) + ')', text))
            for name, text in sources]


xmodel.module_sources = export_sources

_base = xroles.role_of


def role_of(cx, s):
    par = cx.parent.get(s.name)
    how = par[1].kind if par else 'root'
    atoms = s.atom_kinds()
    operations = xroles.lets(s)
    params = dict(how=how)
    if par:
        params['count'] = par[1].count
    if how == 'root' and s.name != 'generatedRoot':
        return 'unused.body', params
    if how == 'seq' and set(atoms) == {'familyGetDynamic'} and \
            operations.get('matrixAdd') == atoms['familyGetDynamic'] and \
            len(s.inputs) == 2 * len(s.output):
        params.update(k=len(s.output), F=xroles.fam_len(s.inputs[len(s.output)]))
        return 'sum.members.step', params
    if how == 'seq' and 'blockDiagonalRuns' in atoms:
        return 'smudge.digit.step', params
    if how == 'par' and set(atoms) == {'familyGetDynamic'} and 'multiplyMonomial' in operations:
        return 'sampler.gather-placement', params
    if how == 'par' and set(atoms) == {'familyGetDynamic'} and len(s.inputs) == 1 and \
            ' × ' not in s.input_type and ' × ' not in s.output_type:
        return 'gather.scalar', params
    if how == 'par' and not atoms and set(operations) == {'liftInteger', 'matrixMulScalarLeft'}:
        return 'smudge.digit.weight', params
    if how == 'par' and not atoms and not s.inputs and len(s.output) == 1 and \
            set(operations) == {'const'}:
        return 'label.zero', params
    if how == 'par' and 'uniformIntervalSample' in atoms and 'gadgetMatrixRuns' in atoms:
        return ('label.forest.branch' if 'ringAutomorphismRuns' in atoms
                else 'label.forest.entry'), params
    if how == 'par' and s.loops and 'select' in atoms and 'familyGetStatic' in atoms:
        return 'label.forest.assemble', params
    if how == 'par' and 'select' in atoms and 'familyGetDynamic' in atoms:
        if 'hashSample' in atoms:
            return 'label.forest.public', params
        if 'gadgetDecomposeRuns' in atoms:
            return 'label.forest.evaluate', params
        if not s.loops:
            return 'label.forest.gather', params
    r, reason = _base(cx, s)
    if r == 'instance.body':
        # A root parallel over the members of one LUT round (its loop child a LUT step): the round
        # at one modulus (no move down), bundled like refresh_top's member roles.
        kids = [(a.loop.kind, cx.sc.get(a.loop.child)) for a in s.loops]
        # A root parallel over the members of a PRF row sum (refresh_top's `prf.rowsum`, which
        # refresh_top only meets inside a part instance).
        A = s.atom_kinds()
        if set(A) == {'familyGetStatic'} and [k for k, _ in kids] == ['par', 'seq'] and \
                A['familyGetStatic'] == 2 * len(s.output) and \
                [role_of(cx, c)[0] for _, c in kids] == ['gather.unpack', 'sum.members.step']:
            loop = [a.loop for a in s.loops][0]
            p = dict(reason)
            p.update(k=len(s.output), W=loop.count, F=xroles.fam_len(s.inputs[0]))
            return 'prf.rowsum', p
        # A root parallel over the members of a LUT key table (the switching ciphertexts of a
        # LUT, encrypted without an error term; refresh_top's `lut.key-table` samples one).
        if [k for k, _ in kids] == ['par', 'seq'] and \
                [role_of(cx, c)[0] for _, c in kids] == ['lut.key-table.index', 'lut.key-table.step'] \
                and len(s.inputs) == 3 and len(s.output) == 4:
            p = dict(reason)
            p.update(W=[a.loop for a in s.loops][1].count)
            return 'lut.key-table.flat', p
        if len(kids) == 1 and kids[0][0] == 'seq' and \
                _base(cx, kids[0][1])[0] in ('lut.round.step', 'lut.key-round.step'):
            p = dict(reason)
            return ('lut.member.flat' if _base(cx, kids[0][1])[0] == 'lut.round.step'
                    else 'lut.key-member.flat'), p
    if r is not None:
        return r, reason
    par = cx.parent.get(s.name)
    how = par[1].kind if par else 'root'
    A = s.atom_kinds()
    keys = set(A)
    L = xroles.lets(s)
    kids = s.loops
    kin, kout = len(s.inputs), len(s.output)
    p = dict(how=how)
    if par is not None:
        p['count'] = par[1].count
    if how == 'par' and not keys and not kids and not L and kin == kout:
        p.update(k=kout)
        return 'import.copy', p
    if how == 'par' and not keys and not kids and set(L) == {'liftInteger', 'multiplyMonomial'} and \
            kin == 4 and kout == 3:
        return 'import.label', p
    if how == 'par' and not keys and not kids and set(L) == {'ofNat', 'multiplyMonomial'} and \
            L['multiplyMonomial'] == kout == kin:
        p.update(k=kout)
        return 'sampler.monomial', p
    if how == 'gather' and not kids and {'hashSample', 'gadgetDecomposeRuns', 'concatRows'} <= keys and \
            'ringAutomorphismRuns' not in keys:
        p.update(k=kout, dynamic='familyGetDynamic' in keys)
        return 'prf.cell-flat', p
    if how == 'par' and not kids and keys == {'preimageRunsDispatched'}:
        return 'preimage.sample', p
    if how == 'par' and not kids and {'hashSample', 'gadgetDecomposeRuns', 'gadgetMatrixRuns'} <= keys:
        p.update(k=kout, dynamic='familyGetDynamic' in keys)
        return 'enc.rooted-branch-key', p
    if how == 'par' and not kids and keys == {'hashSample', 'gadgetMatrixRuns', 'centeredRebaseRuns'} and \
            'liftInteger' in L:
        return 'enc.rooted-label-entry', p
    return None, reason


xroles.role_of = role_of

if __name__ == '__main__':
    args = sys.argv[1:]
    gdir = args.pop(0)
    out = None
    if '--json' in args:
        i = args.index('--json')
        out = args[i + 1]
        del args[i:i + 2]
    modules = args or sorted(f[:-5] for f in os.listdir(gdir)
                             if f.startswith(('Stage_', 'W_Stage_')) and f.endswith('.lean')
                             and '_Facts' not in f and '_Part_' not in f)
    import collections
    import json
    res, unknown = xroles.assign(gdir, modules)
    for m, d in res.items():
        c = collections.Counter(v['role'] for v in d.values())
        print(f'{m}: {len(d)} scopes, roles ' + ', '.join(f'{r}:{n}' for r, n in sorted(c.items())))
    if out:
        json.dump(res, open(out, 'w'), indent=1)
    for m, n, why in unknown:
        print(f'UNKNOWN {m} {n}: {why}')
    print(f'{len(unknown)} unknown')
    sys.exit(1 if unknown else 0)
