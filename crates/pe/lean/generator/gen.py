#!/usr/bin/env python3
"""Write the Lean proof package of an average-case PE parameter set.

Usage: gen.py <parameter id>

Reads the row of `lean/parameters.csv` and the claim the integration test exported to
`lean/<parameter id>/generated`, derives every parameter-dependent constant of the proof (the
modulus, the digit counts, the gadget base, the cutoffs, the widths, and the hash tags), and fills
`generator/templates` into `lean/<parameter id>`. The templates are the proof of
`average-w2-n8192-crt5x30`; they hold for every row of the same class and predicate (one attribute
bit, `F(x) = x₀`), ring dimension `2^13`, two digits per CRT prime, and the average noise model.
Before writing, the script checks the numeric conditions the proof decides (`numeric_margin`,
`margin_lt`) so that a row they reject fails here rather than in Lean.
"""
import csv
import math
import os
import re
import shutil
import sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
LEAN = os.path.dirname(HERE)
N = 8192


def row_of(parameter_id):
    with open(os.path.join(LEAN, 'parameters.csv')) as f:
        rows = [row for row in csv.DictReader(f) if row['parameter_id'] == parameter_id]
    if len(rows) != 1:
        sys.exit(f'{parameter_id}: expected exactly one row in parameters.csv')
    return rows[0]


def read(generated, name):
    with open(os.path.join(generated, name + '.lean')) as f:
        return f.read()


def one(pattern, text, what):
    matches = set(re.findall(pattern, text))
    if len(matches) != 1:
        sys.exit(f'expected one {what}, found {sorted(matches)[:3]}')
    return matches.pop()


def online_bound(d, m, half, b0, bb, bz, bh):
    inner = m * N * b0 * half + m * (d * N * bz * half + d * N * bz * half) + 1 * bb
    return (d * N * inner * half +
            d * N * (m * N * bh * half + (m * N * inner * half + m * (d * N * bz * half + d * N * bz * half) +
                                          1 * bh)) * half)


def values(parameter_id):
    row = row_of(parameter_id)
    if row['noise_model'] != 'average' or not row['failure_exponent'].isdigit():
        sys.exit('the templates prove the average model with a failure exponent')
    # The tail `N 2 exp(-x) <= N 2^(1 - x)` reaches `2^-fail` for `x >= fail + 20` at `N = 2^13`.
    fail = int(row['failure_exponent'])
    tail = fail + 20
    if (row['attribute_length'], row['alphabet_size'], row['state_width'], row['max_steps'], row['reads'],
            row['tables']) != ('1', '2', '1', '1', '0', '0:1'):
        sys.exit('the templates prove the class (1, 2, 1, 1) with the predicate F(x) = x₀')
    if int(row['ring_dimension']) != N:
        sys.exit(f'the templates prove the ring dimension {N}')
    # The templates model the computational TrapGen `[a | I | G - (a R + E)]` (`m_B = D + 2`) and
    # the default cutoffs; a statistical row needs proof templates of its own TrapGen.
    if row['trapdoor_instantiation'] != 'computational' or row['cutoff_multiplier'] != '6.5':
        sys.exit('the templates prove the computational instantiation with the default cutoffs')
    generated = os.path.join(LEAN, parameter_id, 'generated')
    backend = read(generated, 'Backend')
    q = int(one(r'RegularLayout (\d+) :=', backend, 'modulus'))
    moduli = [int(x) for x in one(r'def moduli0 : List Nat := \[([0-9, ]+)\]', backend, 'moduli').split(',')]
    base = int(one(r'base := (\d+)', backend, 'gadget base'))
    if one(r'digitsPerTower := (\d+)', backend, 'digits per tower') != '2':
        sys.exit('the templates prove two digits per CRT prime')
    if len(moduli) != int(row['crt_depth']) or base != 2 ** int(row['base_bits']):
        sys.exit('the exported layout does not match the row')
    d = 2 * len(moduli)
    m = 2 * d
    encrypt, prelude = read(generated, 'Stage_encrypt'), read(generated, 'Stage_prelude')
    keygen, trapdoor = read(generated, 'Stage_keygen'), read(generated, 'Stage_setup_trapdoor')
    preprocess, decode = read(generated, 'Stage_preprocess'), read(generated, 'Stage_decode')
    # The (width, cutoff) pairs of the Gaussian laws of a stage: the prelude samples `chi` only,
    # and encryption adds `chi_fl` unless both laws coincide. Widths and cutoffs of any size are
    # read whole.
    laws = lambda text: set(re.findall(
        r'gaussianSampleAt tape \(path \+\+ \[\d+\]\) \(\((\d+(?:/\d+)?)\)\) \((\d+)\)', text))
    if len(laws(prelude)) != 1:
        sys.exit('expected one prelude Gaussian law')
    (sigma_e, ce), = laws(prelude)
    flooding = laws(encrypt) - laws(prelude)
    if len(flooding) > 1 or sigma_e != '319/100':
        sys.exit('unexpected encryption Gaussian laws')
    ce = int(ce)
    cf = int(flooding.pop()[1]) if flooding else ce
    key_sigma, key_cutoff = one(r'gaussianSampleAt tape \(path \+\+ \[7\]\) \(\((\d+/\d+)\)\) \((\d+)\)', keygen,
                                'key law')
    trap_sigma = one(r'trapdoorSampleAt backend tape \(path \+\+ \[0\]\) \((\d+/\d+)\)', trapdoor, 'TrapGen width')
    dynamic = r'\[114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99\]\) \(\[\.bytes \[([0-9, ]+)\]'
    schema = one(dynamic, preprocess, 'setup schema')
    context = one(r'\[109, 120, 120, 45, 112, 101\]\) \(\[\.bytes \[([0-9, ]+)\]', decode, 'PE context')
    sigma = Fraction(key_sigma)
    digits = 3
    scale = 10 ** max(0, len(str(math.ceil(sigma))) - digits)
    sigma_bound = math.ceil(sigma * Fraction(101, 100) / scale) * scale
    key_cutoff = int(key_cutoff)
    half = base // 2
    mb = d + 2
    bf = online_bound(d, m, half, cf, mb * N * ce * ce, cf, ce)
    bias = mb * N * ce * key_cutoff
    mt = (q - 8) // 4 - (ce + 2) - 2 * bias
    energy = m * N * (ce * ce + ce * ce + bf * bf)
    if not (2 * bf < q and 2 * tail * sigma_bound * sigma_bound * energy <= mt * mt and
            4 * (ce + 1 + 2 * bias + mt) + 4 < q):
        sys.exit(f'{parameter_id}: the bounds of the proof do not reach a failure probability of 2^-{fail}')
    return {
        'ID': parameter_id,
        'FAIL': str(fail), 'TAIL': str(tail), 'TAIL2': str(2 * tail),
        'PACKAGE': 'MxxPe' + ''.join(part.capitalize() for part in parameter_id.split('-')),
        'DEPTH': str(len(moduli)), 'CRT_BITS': row['crt_bits'], 'BASE_BITS': row['base_bits'],
        'QBITS': str(q.bit_length()), 'Q': str(q), 'HALFQ': str(q // 2), 'INV2': str((q + 1) // 2),
        'D': str(d), 'M': str(m), 'L': str(m * m), 'MB': str(mb), 'BASE': str(base), 'HALFBASE': str(half),
        'CE': str(ce), 'CE1': str(ce + 1), 'CE2': str(ce + 2), 'CF': str(cf),
        'KEYSIGMA': key_sigma, 'KEYCUTOFF': str(key_cutoff), 'SIGMABOUND': str(sigma_bound),
        'TRAPSIGMA': trap_sigma.replace('/', ' / '),
        'SCHEMA_BYTES': '[' + schema + ']', 'PECTX_BYTES': '[' + context + ']',
    }, math.log2(bf)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    parameter_id = sys.argv[1]
    subst, log_bf = values(parameter_id)
    out = os.path.join(LEAN, parameter_id)
    templates = os.path.join(HERE, 'templates')
    for name in sorted(os.listdir(templates)):
        with open(os.path.join(templates, name)) as f:
            text = f.read()
        text = re.sub(r'\{\{(\w+)\}\}', lambda match: subst[match.group(1)], text)
        with open(os.path.join(out, name), 'w') as f:
            f.write(text)
    shutil.copy(os.path.join(LEAN, 'lean-toolchain'), os.path.join(out, 'lean-toolchain'))
    print(f'{parameter_id}: wrote {out} (online error bound 2^{log_bf:.1f})')


if __name__ == '__main__':
    main()
