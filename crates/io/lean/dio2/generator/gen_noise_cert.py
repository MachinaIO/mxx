#!/usr/bin/env python3
"""Kernel-checked arithmetic for Dio2Noise, with a fail-closed production preflight.

Usage (from the repository root)::

  python3 crates/io/lean/dio2/generator/gen_noise_cert.py --preset toy --out DIR
  python3 crates/io/lean/dio2/generator/gen_noise_cert.py --preset production --out DIR
  python3 crates/io/lean/dio2/generator/gen_noise_cert.py --params params.json --out DIR

The JSON manifest records the actual primes, quotients, CRT basis, dimensions,
cutoffs and layout. The toy profile mirrors Dio2Online0Assembly0 and Dio2Noise,
including the duplicated arithmetic in Dio2Complete. Every expression evaluation
and every decoder inequality gets its own theorem over closed Nat arithmetic.
No native_decide, external solver, or axiom is used.

With --sharp --metadata FILE, production emits the conservative monomial-cell
certificate (conditional on the symbolic stage premises). Metadata supplies
exported dimensions and routing; the manifest supplies noise cutoffs and PRF P.

IMPORTANT: without --sharp the production profile cannot pass the literal toy norm recurrence.
Even 34 buckets of width ONE, unit imported noise, and ZERO preprocessing noise
exceed every quotient. The generator emits a kernel-checkable obstruction and
returns 2 rather than inventing a successful certificate. This is failure of
these sufficient bounds, not evidence that execution decodes incorrectly.
Production target/level wiring is deliberately not guessed after this failure.
A successful production certificate requires a stronger symbolic stage lemma
(e.g. exploiting monomial multiplication) before extending the arithmetic port.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
TOY_PROOF = ROOT / 'average-n8-p5x50/proof'


def exported_parameters(directory: Path) -> tuple[list[int], int, int]:
    """Read split or unsplit exports, without relying on wire numbers."""
    paths = sorted(directory.glob('Stage_online_0*.lean'))
    primes: dict[int, int] = {}
    gadgets = set()
    modulus = None
    for path in paths:
        source = path.read_text()
        aliases = dict(re.findall(r'local notation "(mxxNat\d+)" => (\d+)', source))
        for index, value in re.findall(r'abbrev modulus_(\d+) : Nat := (\w+)', source):
            value = int(aliases.get(value, value))
            if int(index) == 0:
                if modulus is not None and modulus != value:
                    raise ValueError('inconsistent exported moduli')
                modulus = value
            else:
                if int(index) in primes and primes[int(index)] != value:
                    raise ValueError('inconsistent exported primes')
                primes[int(index)] = value
        gadgets.update((int(b), int(m)) for b, m in re.findall(
            r'gadgetMatrixRuns backend (\d+) (\d+)', source))
    ordered = [primes[i] for i in range(1, len(primes) + 1)]
    if not ordered or math.prod(ordered) != modulus or len(gadgets) != 1:
        raise ValueError('missing or inconsistent exported modulus/gadget data')
    return ordered, *gadgets.pop()


def preset(name: str) -> dict:
    toy = name == 'toy'
    directory = ROOT / ('worst-n8-p5x50' if toy else 'worst-n65536-p27x60') / 'generated'
    primes, base, columns = exported_parameters(directory)
    modulus = math.prod(primes)
    quotients = [modulus // q for q in primes]
    preimages = (directory / 'Stage_preimages_0_0.lean').read_text()
    cutoffs = {int(v) for v in re.findall(r'Int.toNat \((\d+)\)', preimages)}
    if len(cutoffs) != 1:
        raise ValueError('ambiguous preimage cutoff')
    return dict(profile=name, ring_dimension=8 if toy else 65536,
                primes=primes, quotients=quotients,
                crt_basis=[a * pow(a, -1, q) for a, q in zip(quotients, primes)],
                gadget_base=base,
                digits_per_prime=[math.ceil(q.bit_length() / (base.bit_length()-1)) for q in primes],
                columns=columns, anchor_columns=2*(columns+2),
                prf_nu=1 if toy else 1024, prf_h=1 if toy else 31,
                prf_Q=16 if toy else 256, prf_P=8 if toy else 32,
                bucket_widths=[2, 2] if toy else [130]*34,
                levels=6 if toy else 38, smudge_digits=1 if toy else 306,
                encoding_sigma=0 if toy else 4, trapdoor_sigma=4,
                imported_bound=1, switch_bound=0,
                preflight_optimistic=not toy,
                exported_gaussian_cutoff=0 if toy else 26,
                preimage_cutoff=cutoffs.pop(),
                source=str(directory.relative_to(ROOT.parent.parent.parent.parent)))


def validate(p: dict) -> None:
    q0 = math.prod(p['primes'])
    if p['columns'] != sum(p['digits_per_prime']):
        raise ValueError('gadget columns disagree with prime digits')
    if not p['bucket_widths'] or min(p['bucket_widths']) < 1:
        raise ValueError('empty PRF or bucket')
    if len(p['bucket_widths']) != (2 if p['prf_h'] == 1 else p['prf_h'] + 3):
        raise ValueError('bucket count disagrees with PrfSettings::parameters')
    for i, (q, quotient, basis) in enumerate(zip(p['primes'], p['quotients'], p['crt_basis'], strict=True)):
        if quotient*q != q0 or basis % q != 1:
            raise ValueError(f'invalid quotient/basis at prime {i}')
        if any(basis % r for j, r in enumerate(p['primes']) if j != i):
            raise ValueError(f'invalid CRT basis at prime {i}')
    for key in ('ring_dimension', 'preimage_cutoff', 'columns', 'smudge_digits'):
        if p[key] <= 0:
            raise ValueError(f'{key} must be positive')


class Certificate:
    """Small expression DAG: definitions contain numerals, not recursive products."""
    def __init__(self):
        self.values: dict[str, int] = {}
        self.lines = ['import Mathlib.Tactic.NormNum', '',
                      'set_option maxRecDepth 16384',
                      'set_option maxHeartbeats 8000000', '',
                      'namespace Dio2NoiseCertificate', '']
        self.rows = []

    def add(self, name: str, expr: str) -> int:
        # Expressions are constructed below, never evaluated from user input.
        def evaluate(node):
            if isinstance(node, ast.Constant) and type(node.value) is int:
                return node.value
            if isinstance(node, ast.Name):
                return self.values[node.id]
            if isinstance(node, ast.BinOp):
                left, right = evaluate(node.left), evaluate(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.FloorDiv):
                    return left // right
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == 'max' and len(node.args) == 2 and not node.keywords):
                return max(map(evaluate, node.args))
            raise ValueError('unsupported arithmetic expression')
        value = evaluate(ast.parse(expr, mode='eval').body)
        assert isinstance(value, int) and value >= 0
        refs = sorted(set(re.findall(r'\b[A-Za-z]\w*\b', expr)) & self.values.keys())
        lean = expr.replace('//', '/')
        if lean.startswith('max('):
            a, b = lean[4:-1].split(',')
            lean = f'max ({a}) ({b})'
        self.values[name] = value
        unfold = ', '.join([name] + refs)
        self.lines += [f'def {name} : Nat := {value}',
                       f'theorem {name}_eval : {name} = {lean} := by',
                       f'  norm_num [{unfold}]', '']
        return value

    def compare(self, name: str, left: str, right: str, reject=False):
        l, r = self.values[left], self.values[right]
        accepted = l < r
        if not accepted and not reject:
            raise ValueError(f'{name}: decoder inequality failed ({math.log2(l)} >= {math.log2(r)})')
        op = '<' if accepted else '≥'
        self.lines += [f'theorem {name} : {left} {op} {right} := by',
                       f'  norm_num [{left}, {right}]', '']
        self.rows.append(dict(theorem=name, left=left, right=right, accepted=accepted,
                             log2_left=math.log2(l), log2_quotient=math.log2(r)))

    def write(self, out: Path, name: str):
        (out / f'{name}.lean').write_text('\n'.join(self.lines + ['end Dio2NoiseCertificate', '']))
        (out / 'bounds.json').write_text(json.dumps(self.values, indent=2)+'\n')
        (out / 'inequalities.json').write_text(json.dumps(self.rows, indent=2)+'\n')
        (out / 'bits.tsv').write_text('theorem\tlog2_left\tlog2_quotient\taccepted\n' + ''.join(
            f"{r['theorem']}\t{r['log2_left']:.9f}\t{r['log2_quotient']:.9f}\t{r['accepted']}\n"
            for r in self.rows))


def constants(c: Certificate, p: dict):
    for name, key in [('n', 'ring_dimension'), ('m', 'columns'), ('base', 'gadget_base'),
                      ('anchor', 'anchor_columns'), ('cutoff', 'preimage_cutoff')]:
        c.add(name, str(p[key]))
    c.add('half', 'base // 2')
    c.add('action', '(m + m) * n * half')
    c.add('modulus', str(math.prod(p['primes'])))
    for i, q in enumerate(p['quotients']):
        c.add(f'prime{i}', str(p['primes'][i]))
        c.add(f'quotient{i}', str(q))
        c.lines += [f'theorem quotient{i}_exact : quotient{i} * prime{i} = modulus := by',
                    f'  norm_num [quotient{i}, prime{i}, modulus]', '']


def toy_certificate(c: Certificate, p: dict):
    """Literal toy formulas, including target multiplicities reduced to semantic classes."""
    if (p['ring_dimension'], p['columns'], p['gadget_base'], p['bucket_widths'],
            p['smudge_digits'], p['imported_bound'], p['switch_bound']) != (8,25,4096,[2,2],1,1,0):
        raise ValueError('toy target adapters require the exact toy layout')
    c.add('selectorCellBound', 'action * (action * 1 + n * 1 * 1) + n * 1 * 0')
    c.add('selectorBucketBound', '2 * selectorCellBound')
    c.add('readoutInputBound', '2 * (action * (action * 1 + n * 1 * selectorBucketBound) + n * 1 * 0)')
    c.add('sigmaReadout', '7 * (action * 1 + action * readoutInputBound) + (action * 1 + action * readoutInputBound) + 8 * (n * 1 * 0)')
    c.add('sigmaBound', 'n * (7 * (action * 1 + action * sigmaReadout) + (action * 1 + action * sigmaReadout) + 8 * (n * 1 * 0))')
    c.add('gBound', 'm * n * sigmaBound * half')
    c.add('c1Bound', 'm * n * 1 * half')
    c.add('stBound', '3 * (action * 1 + action * selectorCellBound) + (action * 1 + action * selectorCellBound) + 4 * (n * 1 * 0)')
    c.add('pubBound', 'max(m * n * gBound * half, m * n * c1Bound * half)')
    for name in ['freshBound', 'smudgeBound']:
        c.add(name, 'n * (7 * (action * 1 + action * readoutInputBound) + (action * 1 + action * readoutInputBound) + 8 * (n * 1 * 0))')
    c.add('label2', 'action * 1 + n * 1 * 1')
    c.add('label4', 'action * label2 + n * 1 * label2')
    c.add('label3', 'action * label2 + n * 1 * 1')
    for name, product in [('lab425Bound','label2'), ('lab543Bound','label4'),
                          ('lab626Bound','label3'), ('lab680Bound','label2'), ('lab734Bound','label2')]:
        c.add(name, f'm * n * (m * n * {product} * half) * half + m * n * (m * n * {product} * half) * half + n * 1 * pubBound')
    c.add('sigBound', 'max(m * n * (m * n * sigmaBound * half) * half, m * n * c1Bound * half)')
    c.add('autBound', 'max(m * n * (m * n * (action * 1 + action * sigmaBound + n * 1 * 0) * half) * half, m * n * c1Bound * half)')
    c.add('privateBound', 'm * n * 1 * half')
    c.add('stateTargetBound', 'm * n * stBound * half')
    c.add('anchorBound', '0')
    targets = ['privateBound','pubBound','stateTargetBound','lab425Bound','lab543Bound',
               'lab626Bound','lab680Bound','lab734Bound','sigBound','autBound','anchorBound']
    for target in targets:
        c.add(f'left_{target}', f'2 * (anchor * n * 1 * cutoff + (m * n * gBound * half + m * n * c1Bound * half + m * n * {target} * half) + m * n * ((m * m) * n * freshBound * half) * half + (m * m) * n * smudgeBound * half + 4)')
        for i in range(len(p['primes'])):
            c.compare(f'step0_p{i}_{target}', f'left_{target}', f'quotient{i}')
    c.add('targetBound', 'autBound')
    c.add('step0Left', 'left_autBound')
    c.add('expandedInputBound', 'n * 1 * 1 + 1 + 1')
    # 8 is the budget actually used in Complete; 1 and 10 are Noise's alternatives.
    for bound in [1, 8, 10]:
        c.add(f'finalCell{bound}', f'action * (action * {bound} + n * 1 * {bound}) + n * 1 * 0')
        c.add(f'finalBucket0_{bound}', f'2 * finalCell{bound}')
        c.add(f'finalBucket1_{bound}', f'2 * (action * (action * {bound} + n * 1 * finalBucket0_{bound}) + n * 1 * 0)')
        c.add(f'finalState{bound}', f'3 * (action * {bound} + action * finalCell{bound}) + (action * {bound} + action * finalCell{bound}) + 4 * (n * 1 * 0)')
        c.add(f'finalSmudge{bound}', f'n * (7 * (action * {bound} + action * finalBucket1_{bound}) + (action * {bound} + action * finalBucket1_{bound}) + 8 * (n * 1 * 0))')
        c.add(f'finalLeft{bound}', f'2 * (anchor * n * {bound} * cutoff + m * n * finalState{bound} * half + m * n * finalSmudge{bound} * half + 4)')
        c.compare(f'step1_bound{bound}', f'finalLeft{bound}', 'quotient0')


def toy_regression(c: Certificate, out: Path):
    """Diff against numerals extracted from the checked-in proof, not Python fixtures."""
    source = (TOY_PROOF / 'Dio2Noise.lean').read_text()
    expected = {}
    for name in ['sigmaBound','gBound','c1Bound','stBound','pubBound','freshBound',
                 'smudgeBound','lab425Bound','lab543Bound','lab626Bound','lab680Bound',
                 'lab734Bound','sigBound']:
        field = 'σBound' if name == 'sigmaBound' else name
        expected[name] = int(re.search(r'inputs\.' + field + r' = (\d+)', source)[1])
    for name in ['targetBound','step0Left']:
        expected[name] = int(re.search(r'def '+name+r' : Nat :=\s*(\d+)', source)[1])
    for name, expr in [('selectorCellBound','selectorCellBound 1'),
                       ('selectorBucketBound','selectorBucketBound 1'),
                       ('readoutInputBound','readoutInputBound')]:
        expected[name] = int(re.search(re.escape(expr)+r' = (\d+)', source)[1])
    for bound in [1,10]:
        for name in ['State','Smudge']:
            expected[f'final{name}{bound}'] = int(re.search(f'final{name}Bound {bound} = (\\d+)', source)[1])
    for name, expr in [('finalCell10', 'finalCellBound 10'),
                       ('finalBucket0_10', 'finalBucket0Bound 10'),
                       ('finalBucket1_10', 'finalBucket1Bound 10')]:
        expected[name] = int(re.search(re.escape(expr)+r' = (\d+)', source)[1])
    expected['finalLeft10'] = int(re.search(r'theorem final_left_value[\s\S]*?=\s*(\d+) :=', source)[1])
    complete = (TOY_PROOF / 'Dio2Complete.lean').read_text()
    for name in ['sigmaBound','gBound','c1Bound','stBound','pubBound','freshBound','smudgeBound','targetBound','step0Left']:
        if str(expected[name]) not in complete:
            raise ValueError(f'{name} missing from duplicated Complete arithmetic')
    actual = {name:c.values[name] for name in expected}
    (out/'toy_expected.json').write_text(json.dumps(expected,indent=2)+'\n')
    (out/'toy_actual.json').write_text(json.dumps(actual,indent=2)+'\n')
    if expected != actual:
        raise ValueError('toy regression differs; compare toy_expected.json and toy_actual.json')
    print(f'Toy regression: {len(expected)} exact numerals match Dio2Noise; duplicated Complete numerals match.')


def production_preflight(c: Certificate, p: dict) -> bool:
    """A smaller recurrence than the requested one suffices to refute acceptance.

    Nonnegative helper/label/switch terms may be discarded. Width one is a
    lower bound for every nonempty bucket. Keeping the label term at one is
    still below production imported noise. This avoids assuming an unported
    production LUT or label adapter while exposing the numerical obstruction.
    """
    for tag, widths in [('optimistic', [1]*len(p['bucket_widths'])),
                        ('literal', p['bucket_widths'])]:
        c.add(f'{tag}Seed', '1')
        previous = f'{tag}Seed'
        for i, width in enumerate(widths):
            name = f'{tag}Bucket{i}'
            c.add(name, f'{width} * (action * (action * 1 + n * 1 * {previous}) + n * 1 * 0)')
            previous = name
        # Even the bucket itself is above Q/q: all later toy-formula readout,
        # packing and decoder actions have positive integer factors >= 1.
        c.add(f'{tag}DecoderLower', f'2 * {previous}')
        for i in range(len(p['primes'])):
            c.compare(f'{tag}_obstruction_p{i}', f'{tag}DecoderLower', f'quotient{i}', reject=True)
            if c.values[f'{tag}DecoderLower'] >= p['quotients'][i]:
                c.lines += [
                    f'theorem {tag}_no_decoder_p{i} (E : Nat) (h : {previous} ≤ E) :',
                    f'    ¬ 2 * E < quotient{i} := by',
                    f'  have hcut := {tag}_obstruction_p{i}',
                    f'  rw [{tag}DecoderLower_eval] at hcut',
                    '  exact Nat.not_lt.mpr (le_trans hcut (Nat.mul_le_mul_left 2 h))', '']
    c.add('singleDigitResidual', str(p['prf_P']//2))
    # Odd digits +/- (P-1), weights P^r/2, and ONE final half offset:
    # ((P-1) * sum(P^r) + 1) / 2 = P^d / 2 (dio2/noise.rs).
    c.add('multiDigitResidual', str(p['prf_P']**p['smudge_digits']//2))
    return all(r['accepted'] for r in c.rows)



def sharp_parameters(p: dict, metadata: Path | None) -> dict:
    """Overlay extracted geometry; reject disagreement with the supplied manifest.

    Without metadata, the caller must supply label_columns and members_per_column
    in the fallback manifest. This makes fallback assumptions reviewable.
    """
    p = dict(p)
    if metadata is not None:
        data = json.loads(metadata.read_text())
        v = data['views']['generated']
        layout, = [l for l in data['backend_layouts'] if l['modulus'] == v['Q0']]
        shapes = list(v['prfs'].values())
        widths = shapes[0]['bucket_widths']
        if any(s['bucket_widths'] != widths for s in shapes):
            raise ValueError('sharp envelope requires a common PRF shape')
        members = {s['members'] // s['packed_members'] for s in shapes
                   if s['packed_members'] and s['members'] % s['packed_members'] == 0}
        if members != {v['ring_dimension']}:
            raise ValueError('inconsistent members per packed column')
        geometry = dict(ring_dimension=v['ring_dimension'], primes=v['primes'],
                        quotients=[v['Q0'] // q for q in v['primes']],
                        columns=v['columns'], anchor_columns=v['anchor_columns'],
                        gadget_base=layout['base'],
                        digits_per_prime=[layout['digitsPerTower']] * len(v['primes']),
                        bucket_widths=widths, smudge_digits=v['smudge_digits'],
                        levels=v['level_count'])
        for key, value in geometry.items():
            if key in p and p[key] != value:
                raise ValueError(f'manifest/metadata mismatch: {key}')
        p.update(geometry)
        p.update(label_columns=v['labels']['columns'], members_per_column=members.pop(),
                 lut_branches=max(len(t['values']) for ts in v['lut_tables'].values() for t in ts),
                 target_classes=v['target_classes'], metadata_source=str(metadata),
                 metadata_sources=data['sources'])
        kinds = {t['kind'] for t in v['target_classes']}
        if kinds != {'private', 'public', 'state', 'labels', 'sigma', 'Aut', 'anchor'}:
            raise ValueError(f'unsupported target classes: {kinds}')
    else:
        for key in ['label_columns', 'members_per_column', 'lut_branches']:
            if key not in p:
                raise ValueError(f'sharp fallback manifest requires {key}')
    if p['switch_bound'] != 0 or p['prf_P'] % 2:
        raise ValueError('sharp recurrence requires no modulus drop and even P')
    return p


def sharp_certificate(c: Certificate, p: dict):
    """Exactly a3's conservative envelope; only the PRF public gain is sharpened.

    Definitions b<root>_<stage> are closed numerals; each has an _eval theorem.
    Bucket and label-tree recurrences are exposed one level at a time. The seven
    target envelopes cover both components and all members, not a sum of targets.
    Semantic monomial premises and stage-to-envelope proofs remain external.
    """
    for name, expr in [('A', 'action'), ('C', 'm * n * half'),
                       ('H', 'm * half'), ('D', 'm * m * n * half'),
                       ('F', str(p['exported_gaussian_cutoff'])),
                       ('members', str(p['members_per_column'])),
                       ('branches', str(p['lut_branches'])),
                       ('digits', str(p['smudge_digits']))]:
        c.add(name, expr)
    # ceil(log2(columns+1)) for a balanced product including label-0.
    depth = p['label_columns'].bit_length()
    for root in dict.fromkeys([p['imported_bound'], p['exported_gaussian_cutoff'],
                               p['ring_dimension'] + 2]):
        prefix = f'b{root}_'
        def add(name, expr):
            return c.add(prefix + name, expr)
        def ref(name):
            return prefix + name
        add('root', str(root))
        previous = ref('root')
        for k, width in enumerate(p['bucket_widths'], 1):
            name = f'bucket{k}'
            add(name, f'{width} * (A * (A * {ref("root")} + {previous}) + n * F)')
            previous = ref(name)
        add('bucket', previous)
        add('helper', f'max({ref("root")}, F)')
        add('readout', f'branches * (A * {ref("helper")} + A * {ref("bucket")} + n * F)')
        add('sample', f'branches * (A * {ref("helper")} + A * {ref("readout")} + n * F)')
        add('packed', f'members * {ref("sample")}')
        add('state', ref('packed'))
        add('g', f'C * {ref("packed")}')
        add('one', f'C * {ref("root")}')
        add('public', f'C * {ref("g")}')
        previous = ref('root')
        for k in range(1, depth + 1):
            add(f'labelLevel{k}', f'(A + n) * {previous}')
            previous = ref(f'labelLevel{k}')
        add('labelProduct', previous)
        add('label', f'2 * C * C * {ref("labelProduct")} + n * {ref("public")}')
        add('automorphism', f'C * C * (A * {ref("helper")} + A * {ref("packed")} + n * F)')
        add('private', f'C * {ref("root")}')
        add('stateTarget', f'C * {ref("state")}')
        add('sigma', ref('public'))
        add('anchor', '0')
        add('fresh', f'C * D * {ref("packed")}')
        add('smudge', f'digits * D * {ref("packed")}')
        add('hint', f'anchor * n * {ref("root")} * cutoff')
        add('residue', str(p['prf_P'] ** p['smudge_digits'] // 2))
        add('final', f'{ref("hint")} + C * {ref("state")} + {ref("smudge")}')
        add('left_final', f'2 * ({ref("final")} + {ref("residue")})')
        c.compare(prefix + 'final_p0', ref('left_final'), 'quotient0')
        for target in ['private', 'stateTarget', 'public', 'label', 'sigma', 'automorphism', 'anchor']:
            add('error_' + target, f'{ref("hint")} + C * ({ref("g")} + {ref("one")} + {ref(target)}) + {ref("fresh")} + {ref("smudge")}')
            add('left_' + target, f'2 * ({ref("error_" + target)} + {ref("residue")})')
            for j in range(len(p['primes'])):
                c.compare(prefix + f'{target}_p{j}', ref('left_' + target), f'quotient{j}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--preset', choices=['toy','production'])
    group.add_argument('--params', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--sharp', action='store_true', help='use a3 monomial-cell envelope')
    parser.add_argument('--metadata', type=Path, help='dio2_meta.py JSON (sharp mode only)')
    args = parser.parse_args()
    if args.metadata and not args.sharp:
        parser.error('--metadata requires --sharp')
    p = preset(args.preset) if args.preset else json.loads(args.params.read_text())
    if args.sharp:
        if p['profile'] == 'toy':
            parser.error('--sharp requires production parameters; toy mode is unchanged')
        p = sharp_parameters(p, args.metadata)
    validate(p)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'parameters.json').write_text(json.dumps(p,indent=2)+'\n')
    c = Certificate()
    constants(c,p)
    if p['profile'] == 'toy':
        toy_certificate(c,p)
        toy_regression(c,args.out)
        c.write(args.out,'ToyNoiseCertificate')
        return 0
    if args.sharp:
        sharp_certificate(c,p)
        c.write(args.out,'SharpNoiseCertificate')
        worst = max(r['log2_left'] for r in c.rows)
        margin = min(r['log2_quotient'] - r['log2_left'] for r in c.rows)
        print(f'Sharp: {len(c.rows)} inequalities; max log2(left)={worst:.9f}; min margin={margin:.9f} bits')
        return 0
    passed = production_preflight(c,p)
    c.write(args.out,'ProductionNoiseObstruction')
    print('Production preflight: '+('inconclusive; target adapters still required' if passed else
          'FAILED: even optimistic toy PRF bounds exceed all decoder quotients.'))
    for tag in ['optimistic','literal']:
        print(f"{tag}: log2(2 * bucket) = {math.log2(c.values[tag+'DecoderLower']):.9f}; "
              f"log2(min Q/q) = {math.log2(min(p['quotients'])):.9f}")
    return 2


if __name__ == '__main__':
    sys.exit(main())
