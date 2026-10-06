"""LwrLut.lean: the readout LUT from `Q2` to `Q1`."""
import re
from lean_patterns import proj
from gen_blocks import small


def lut(I):
    return transform(I, 'LwrLut', 'set_option maxHeartbeats 64000000 in\n/-- The readout LUT and its finalization')


def transform(I, name, anchor, extra=()):
    """The template `name` of `average-n32-crt7x40` for the set: the names of the online
    evaluation root after `anchor` are renumbered and reprojected, and the counts replaced."""
    L, B, C, M, N, W, s = I.L, I.B, I.C, I.M, I.n, I.W, I.stride
    m1 = I.m(1)
    t = small(name)
    shift = 22 * (B - 5)
    a = t.index(anchor)
    head, rnd = t[:a], t[a:]
    # Root-level names of the readout round.
    rnd = re.sub(r'\b(w\.w_|h)(\d+)(?=_)', lambda m: m.group(1) + str(int(m.group(2)) + shift)
                 if int(m.group(2)) >= 130 else m.group(0), rnd)
    # The projections of the root.
    lines = rnd.split('\n')
    names = [re.match(r'  have (h\d+_\d+(?:_\d+)?) := hb\.', l).group(1) for l in lines
             if re.match(r'  have h\d+_\d+(?:_\d+)? := hb\.', l)]
    haves = proj(f'{I.d}/generated/Stage_online_eval.lean', 'generatedRoot', names)
    first = next(i for i, l in enumerate(lines) if re.match(r'  have h\d+_\d+(?:_\d+)? := hb\.', l))
    last = max(i for i, l in enumerate(lines) if re.match(r'  have h\d+_\d+(?:_\d+)? := hb\.', l))
    rnd = '\n'.join(lines[:first]) + '\n' + haves.rstrip('\n') + '\n' + '\n'.join(lines[last + 1:])
    # The inputs of the online evaluation root.
    if '    {r00 r01' in rnd:
        old_bind = rnd[rnd.index('    {r00 r01'):rnd.index('{outputs}')]
        fams_ = [f'{{r{j}0 r{j}1 : Fin {C} → ExactMatrix Q{L - 1 - j} N 1 {I.m(L - 2 - j)}}}' for j in range(B)]
        rfams = '\n    '.join(' '.join(fams_[k:k + 2]) for k in range(0, B, 2))
        rnd = rnd.replace(old_bind, '    ' + rfams + '\n    ')
    else:
        old = 'r00 r01 r10 r11 r20 r21 r30 r31 r40 r41}'
        assert old in rnd
        rnd = rnd.replace(old, ' '.join(f'r{j}0 r{j}1' for j in range(B)) + '}')
    ins_ = [f'r{j}0, r{j}1' for j in range(B)]
    rnd = rnd.replace('r00, r01, r10, r11, r20, r21, r30, r31,\n        r40, r41, ()',
                      ',\n        '.join(', '.join(ins_[k:k + 4]) for k in range(0, B, 4)) + ', ()')
    out = head + rnd
    import re as _re
    claim = open(f'{I.d}/generated/Stage_online_eval.lean').read()
    scale = _re.findall(rf'matrixPolynomial \[\((\d+)\)\] : Mxx\.Primitives\.ExactMatrix {I.Q[1]} ', claim)
    assert len(set(scale)) == 1, scale
    reps0 = [
        ('/-- The scale `k` of the finalized readout: `⌊Q1 / 8⌋` with its low `29` bits cleared. -/',
         f'/-- The scale `k` of the finalized readout, near `Q1 / {4 * I.pprime}` with sparse digits. -/'),
        ('abbrev outScale : Int := 137170518016', f'abbrev outScale : Int := {scale[0]}'),
        ('the scale-up `G⁻¹(k)` has two digits of at most', f'the scale-up `G⁻¹(k)` has {m1} digits of at most'),
        ('2 * lutBound B * 524288', f'{m1} * lutBound B * 524288'),
        ('`2^19`', f'`2^{I.digit.bit_length() - 1}`'),
    ]
    for x, y in reps0:
        if name == 'LwrLut':
            assert x in out, x
        out = out.replace(x, y)
    for x, y in extra:
        if x.startswith('re:'):
            out = re.sub(x[3:], y, out)
        else:
            out = out.replace(x, y)
    reps = [
        ('Fin 17 → ExactMatrix Q7 N 1 14', f'Fin {M} → ExactMatrix Q{L} N 1 {I.m(L)}'),
        ('ExactMatrix Q7 N 1 12', f'ExactMatrix Q{L} N 1 {I.m(L - 1)}'),
        ('Fin 17 → ExactMatrix Q7 N 1 1', f'Fin {M} → ExactMatrix Q{L} N 1 1'),
        ('lutA model key Q2 2 5', f'lutA model key Q2 {m1} {B}'),
        ('ExactMatrix Q2 N 1 2', f'ExactMatrix Q2 N 1 {m1}'), ('ExactMatrix Q1 N 1 2', f'ExactMatrix Q1 N 1 {m1}'),
        ('LoopState Q2 Q1 2', f'LoopState Q2 Q1 {m1}'), ('ErrorMatrix N 1 2', f'ErrorMatrix N 1 {m1}'),
        ('reduceMatrix Q1 N 1 2', f'reduceMatrix Q1 N 1 {m1}'),
        ('1048576 2 g', f'{I.base} {m1} g'), ('(m := 2)', f'(m := {m1})'), ("(m' := 2)", f"(m' := {m1})"), ('(log := 5)', f'(log := {I.log_n})'),
        ('(2 + 2) * N', f'({m1} + {m1}) * N'), ('524288', str(I.digit)), ('1048576', str(I.base)),
        ('Xe (2 * z)', f'Xe ({s} * z)'), ('polyNorm_substitution_Xe i (2 * z)', f'polyNorm_substitution_Xe i ({s} * z)'),
        ('moved down to level `8`', 'moved down to `Q1`'), ('Xe (n := N) (2 * z)', f'Xe (n := N) ({s} * z)'),
        ('X^(2z)', f'X^({s}z)'), ('⌊z/16⌋ τ(z mod 16)', f'⌊z/{W}⌋ τ(z mod {W})'),
        ('Fin 16', f'Fin {W}'), ('i < 16', f'i < {W}'), ('i < 15', f'i < {W - 1}'), ('≤ 15', f'≤ {W - 1}'),
        ('(count := 15)', f'(count := {W - 1})'), ('(k := 15)', f'(k := {W - 1})'), ('hhelp 15', f'hhelp {W - 1}'),
        ('hhelp15', 'hhelpLast'), ('(15 : Fin', f'({W - 1} : Fin'), ('sigma_odd 15', f'sigma_odd {W - 1}'),
        ('Fin.last 15', f'Fin.last {W - 1}'), (') = 15 from rfl]', f') = {W - 1} from rfl]'),
        ('15 * lutBranchBound B', f'{W - 1} * lutBranchBound B'), ('16 * (N * 1 * 26)', f'{W} * (N * 1 * 26)'),
        ('(W := 16)', f'(W := {W})'), ('branch `15`', f'branch `{W - 1}`'),
        ('fifteen branches', f'{W - 1} branches'), ('Fifteen branches', f'{W - 1} branches'),
        ('the sixteenth', f'the last'), ('The sixteenth branch', 'The last branch'),
        ('sixteen branches', f'{W} branches'), ('import LwrBlock4', f'import LwrBlock{B - 1}'),
        ('sequential_generatedRoot_131', f'sequential_generatedRoot_{42 + 22 * (B - 1) + 1}'),
        ('hQ7', f'hQ{L}'), (' 26', f' {I.cutoff}'),
    ]
    for x, y in reps:
        out = out.replace(x, y)
    out = out.replace('\nopen Mxx.Primitives MxxRuntime MxxExponentLut\n',
                      '\nopen Mxx.Primitives MxxRuntime MxxExponentLut\n\nset_option maxRecDepth 100000\n', 1)
    # Leftover template counts, unless the counts are the template's.
    leftover = re.findall(r'Fin 17\b|Fin 14\b|\b15\b', out)
    assert (M, C, W) == (17, 14, 16) or not leftover, leftover[:10]
    return out
