"""Private multiplications of the online root: the atoms of one multiplication of a stacked
encoding by a preprocessed RHS, and the segment of the binary offset's two multiplications."""
import re
import sys

from segment import segment
from layouts import layouts

WIRE = re.compile(r'\bw_\d+_(?:\d+|concat_\d+)\b')


def private_mult(named, atom, lets, both_atom):
    """The atoms of a private multiplication of the stacked encoding built by `both_atom`
    (`concatColumns l r both`): the stacking, and per column block `j` the RHS top and bottom, the
    decomposition, the slices and their move down."""
    def by_out(wire):
        return next(x for x in named if re.search(rf'\b{wire}$', atom[x]))
    def out_of(x):
        return WIRE.findall(atom[x])[-1]
    def args_of(x):
        return WIRE.findall(atom[x])[:-1]
    l, r = args_of(both_atom)
    hl = by_out(l); l01, a10 = args_of(hl); hl01 = by_out(l01); c0, a00 = args_of(hl01)
    hr = by_out(r); r01, a11 = args_of(hr); hr01 = by_out(r01); c1, a01 = args_of(hr01)
    both = out_of(both_atom)
    prods = [k for k, v in lets.items() if re.match(rf'MxxRuntime\.matrixMul {both} ', v)]
    prods.sort(key=lambda x: int(x.split('_')[1]))
    out = {'enc': [hl01, hl, hr01, hr, both_atom], 'c': [c0, c1], 'A': [[a00, a01], [a10, a11]],
           'Xb': [], 'D1': [], 'D': [], 'top': [], 'bot': [], 'X': [], 'M0': [], 'M1': [],
           'c2': [], 'A0': [], 'A1': [], 'prods': prods}
    for prod in prods:
        dwire = re.match(rf'MxxRuntime\.matrixMul {both} (\S+)', lets[prod]).group(1)
        hD = by_out(dwire); hD1 = by_out(args_of(hD)[0]); hXb = by_out(args_of(hD1)[0])
        top, bot = args_of(hXb)
        out['Xb'].append(hXb); out['D1'].append(hD1); out['D'].append(hD)
        out['top'].append(top); out['bot'].append(bot)
        def slice_of(a, b):
            return next(x for x in named
                        if re.match(rf'MxxRuntime\.sliceMatrix {prod} {a} {b} 0 ', atom[x]))
        def switch_of(x):
            return next(y for y in named if re.match(
                rf'MxxRuntime\.blockModSwitchRuns \[[^\]]*\] \(1\) {out_of(x)} ', atom[y]))
        for key, (a, b), key2 in [('X', (0, 1), 'c2'), ('M0', (1, 2), 'A0'), ('M1', (2, 3), 'A1')]:
            sl = slice_of(a, b)
            out[key].append(sl)
            out[key2].append(switch_of(sl))
    return out


def private_call(pm, wire):
    """The arguments of `mul_private_down` before `hrhs` and after `hμ`."""
    two = lambda key: f"(fun j ↦ by fin_cases j; exacts [{pm[key][0]}, {pm[key][1]}])"
    w2 = lambda key: f"![{wire(pm[key][0])}, {wire(pm[key][1])}]"
    before = '\n    '.join([
        ' '.join(pm['enc']),
        f"(top := ![{pm['top'][0]}, {pm['top'][1]}]) (bot := ![{pm['bot'][0]}, {pm['bot'][1]}])",
        f"(Xb := {w2('Xb')}) {two('Xb')}",
        f"(D' := {w2('D1')}) {two('D1')}",
        f"(D := {w2('D')}) {two('D')}"])
    after = '\n    '.join([
        f"(X := {w2('X')}) (M := ![![{wire(pm['M0'][0])}, {wire(pm['M0'][1])}], "
        f"![{wire(pm['M1'][0])}, {wire(pm['M1'][1])}]])",
        f"{two('X')} {two('M0')} {two('M1')}",
        f"(c' := {w2('c2')}) (A' := ![![{wire(pm['A0'][0])}, {wire(pm['A0'][1])}], "
        f"![{wire(pm['A1'][0])}, {wire(pm['A1'][1])}]])",
        f"{two('c2')}",
        f"(fun k j ↦ by fin_cases k <;> fin_cases j; exacts [{pm['A0'][0]}, {pm['A0'][1]}, "
        f"{pm['A1'][0]}, {pm['A1'][1]}])"])
    return before, after


def root_offset(path, ns, names, atoms, lets, ab, first_both):
    L = layouts(sys.argv[1])
    """The two private multiplications of the encoding of one by the unit offset RHS (`Q7 → Q6 →
    Q5`), the input of the offset's readout LUT."""
    atom = dict(zip(names, atoms))
    named = [x for x in names if x != '-']
    wire = lambda x: WIRE.findall(atom[x])[-1]
    pm1 = private_mult(named, atom, lets, first_both)
    hl01b = next(y for y in named if atom[y].startswith(f'MxxRuntime.concatRows {wire(pm1["c2"][0])} '))
    hlb = next(y for y in named if atom[y].startswith(f'MxxRuntime.concatRows {wire(hl01b)} '))
    second = next(y for y in named if atom[y].startswith(f'MxxRuntime.concatColumns {wire(hlb)} '))
    pm2 = private_mult(named, atom, lets, second)
    used = []
    for pm in (pm1, pm2):
        for key in ('enc', 'Xb', 'D1', 'D', 'X', 'M0', 'M1', 'c2', 'A0', 'A1'):
            used += pm[key]
    used = [x for x in named if x in used]
    binders, _ = segment(path, ns, 'generatedRoot', used, ab, pm1['top'] + pm1['bot'] + pm2['top'] + pm2['bot'])
    eqs = [b.split(' : ')[0][1:] for b in binders if b.startswith('(e_')]
    c, A = pm1['c'], pm1['A']
    w2 = lambda pm, key: f"![{wire(pm[key][0])}, {wire(pm[key][1])}]"
    a2 = lambda pm: (f"![![{wire(pm['A0'][0])}, {wire(pm['A0'][1])}], "
                     f"![{wire(pm['A1'][0])}, {wire(pm['A1'][1])}]]")
    bl = '\n    '.join(binders)
    s1, t1 = private_call(pm1, wire)
    s2, t2 = private_call(pm2, wire)
    subst = f'  subst {" ".join(eqs)}\n' if eqs else ''
    return f"""
set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The encoding of one moved to `Q5` by two private multiplications with the unit offset RHS:
the input of the binary offset's readout LUT. -/
theorem rootOffset {{model : HashModel}}
    {bl}
    {{sm sp : ErrorPoly N}} (hsm : polyNorm sm ≤ 1)
    {{g6' : ExactMatrix Q6 N 1 12}} {{g7 : ExactMatrix Q7 N 1 12}}
    (hg6' : gadgetMatrixRuns Backend.backend 32768 12 g6') (hg7 : centeredRebaseRuns g6' g7)
    {{B : Nat}}
    (hinput : Rel sm ![![{A[0][0]}, {A[0][1]}], ![{A[1][0]}, {A[1][1]}]]
      (fun j ↦ reducePoly Q7 N (((p6 : Nat) : Int) * 1 * unitPair sp j) • g7) ![{c[0]}, {c[1]}] B)
    {{g5' : ExactMatrix Q5 N 1 10}} {{g6 : ExactMatrix Q6 N 1 10}}
    (hg5' : gadgetMatrixRuns Backend.backend 32768 10 g5') (hg6 : centeredRebaseRuns g5' g6)
    {{eB : Fin 2 → ErrorMatrix N 1 10}} {{BB : Nat}}
    (hrhs : ∀ j, reducePoly Q6 N sp • ![{pm1['top'][0]}, {pm1['top'][1]}] j +
        ![{pm1['bot'][0]}, {pm1['bot'][1]}] j =
      reducePoly Q6 N (((p5 : Nat) : Int) * 1 * unitPair sp j) • g6 + reduceMatrix Q6 N 1 10 (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB)
    {{h5 : ExactMatrix Q5 N 1 8}}
    {{eB' : Fin 2 → ErrorMatrix N 1 8}} {{BB' : Nat}}
    (hrhs' : ∀ j, reducePoly Q5 N sp • ![{pm2['top'][0]}, {pm2['top'][1]}] j +
        ![{pm2['bot'][0]}, {pm2['bot'][1]}] j =
      reducePoly Q5 N (((p4 : Nat) : Int) * Xe (n := N) (2 * 0) * unitPair sp j) • h5 +
        reduceMatrix Q5 N 1 8 (eB' j))
    (heB' : ∀ j, CoeffBound (eB' j) BB') :
    Rel sm {a2(pm2)}
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) * Xe (n := N) (2 * 0) * unitPair sp j) • h5)
      {w2(pm2, 'c2')}
      ((((10 + 10) * N * ((((12 + 12) * N * B * 16384 + (N + 2) * (p6 / 2)) / p6 + N * 1 * BB)) *
        16384 + (N + 2) * (p5 / 2)) / p5 + N * 1 * BB')) := by
{subst}  have hone : polyNorm (1 : ErrorPoly N) ≤ 1 := by simpa using polyNorm_Xe hN1 0
  have h1 := mul_private_down (q := Q7) (q' := Q6) (p := p6) (m := 12) (m' := 10) (bound := 16384)
    hN q7_eq (by norm_num) (by norm_num) (by norm_num) Backend.{L['Q6']} layoutQ6 rfl hg6' hg7 hsm
    hinput
    {s1}
    hrhs heB hone
    {t1}
  have h1' := h1.target_eq (t' := fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * 1 * unitPair sp j) • g6) (by
    funext j
    rw [map_one, one_smul])
  have h2 := mul_private_down (q := Q6) (q' := Q5) (p := p5) (m := 10) (m' := 8) (bound := 16384)
    hN q6_eq (by norm_num) (by norm_num) (by norm_num) Backend.{L['Q5']} layoutQ5 rfl hg5' hg6 hsm h1'
    {s2}
    hrhs' heB' hone
    {t2}
  convert h2 using 2 with j
  rw [map_one, one_smul]
"""
