#!/usr/bin/env python3
"""LwrH2Model.lean: the model in the form of the H2 decomposition.

Usage: gen_h2_model.py <parameter dir>

Writes the pieces of every block and of the readout LUT as functions of the tape, the weights
pulled back from the decoded coefficient, the H2 sites with their coefficients and values, and the
decomposition of the decoded coefficient into the remainder and the sum over the sites.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from info import Info  # noqa: E402
from gen_blocks_e import basis  # noqa: E402


def V(I, b):
    """The weight on the error entering block `b` (the readout LUT for `b = B`) at `tape`."""
    return f'V{b} model key tape' if b == I.B else f'V{b} model key ext tape'


def block_pieces(I, b):
    L, C = I.L, I.C
    qb = L - b
    m = I.m
    mT, mA, ms, mo = m(L), m(qb), m(qb - 1), m(qb - 2)
    P = f'P{qb}'
    c = f'⟨coords {b} i, coords_lt {b} i⟩'
    top = f'(rhsTop model key Q{qb - 1} {mo} rhsPrefix{b} i)'
    bot = f'(bot{b} model key ext tape i)'
    keyIn = '(stateKey model key)' if b == 0 else f'(key{b} model key ext tape)'
    out = [f'/-! ## Block {b} -/\n']
    out.append(f'''noncomputable def eaB{b} (tape : SampleTape) (i : Fin {C}) : Fin 2 → ErrorMatrix N 1 {mT} :=
  memberError tape [3] {c}
''')
    if b == 0:
        out.append(f'''noncomputable def DsB0 (j' : Fin 2) : ErrorMatrix N ({mA} + {mA}) {ms} :=
  liftMatrix (pubDigits0 (stateKey model key) j')
''')
    else:
        A = f'A{qb}'
        out.append(f'''noncomputable def HB{b} : ErrorMatrix N {mT} {mA} := liftMatrix alignDigits{b}

noncomputable def RB{b} (tape : SampleTape) (i : Fin {C}) (k : Fin 2) : ErrorMatrix N 1 {mA} :=
  downResidue {A} (memberRow model key ext tape {c} k * alignDigits{b})

noncomputable def K0B{b} (i : Fin {C}) (k : Fin 2) : ErrorMatrix N 1 {mA} :=
  downResidue {A} (memberKey model key ((coords {b} i : Nat) : Int) 0 k * alignDigits{b})

noncomputable def K1B{b} (i : Fin {C}) (k : Fin 2) : ErrorMatrix N 1 {mA} :=
  downResidue {A} (memberKey model key ((coords {b} i : Nat) : Int) 1 k * alignDigits{b})

noncomputable def DsB{b} (tape : SampleTape) (j' : Fin 2) : ErrorMatrix N ({mA} + {mA}) {ms} :=
  liftMatrix (pubDigits{b} (key{b} model key ext tape) j')
''')
    if b == 0:
        crow = f'(memberRow model key ext tape {c})'
    else:
        crow = (f"(fun j' ↦ theOutput (blockModSwitchRuns {basis(I, L)} 1) "
                f"(memberRow model key ext tape {c} j' * alignDigits{b}))")
    out.append(f'''noncomputable def αB{b} (i : Fin {C}) : ErrorPoly N := alphaOf ext {c}

noncomputable def DrB{b} (tape : SampleTape) (i : Fin {C}) (j : Fin 2) : ErrorMatrix N ({ms} + {ms}) {mo} :=
  liftMatrix (rhsDigits{b} {top} {bot} j)

noncomputable def XB{b} (tape : SampleTape) (i : Fin {C}) (j : Fin 2) : ErrorMatrix N 1 {mo} :=
  downResidue {P} (pairMul (fun j' ↦ pairMul {crow}
      (pubDigits{b} {keyIn} j') +
    matrixMulScalarRight (stateRow{b} model key ext tape j')
      (fun _ _ ↦ reducePoly Q{qb} N (alphaOf ext {c})))
    (rhsDigits{b} {top} {bot} j))

noncomputable def M0B{b} (tape : SampleTape) (i : Fin {C}) (j : Fin 2) : ErrorMatrix N 1 {mo} :=
  downResidue {P} (cellKeyUp{b} (memberKey model key ((coords {b} i : Nat) : Int)) {keyIn}
    {top} {bot} 0 j)

noncomputable def M1B{b} (tape : SampleTape) (i : Fin {C}) (j : Fin 2) : ErrorMatrix N 1 {mo} :=
  downResidue {P} (cellKeyUp{b} (memberKey model key ((coords {b} i : Nat) : Int)) {keyIn}
    {top} {bot} 1 j)

noncomputable def ceB{b} (i : Fin {C}) : ErrorPoly N := alphaOf ext {c} * muOf ext {b}
''')
    ds = '(DsB0 model key)' if b == 0 else f'(DsB{b} model key ext tape)'
    es = '(stateError tape [3])' if b == 0 else f'(err{b} model key ext tape)'
    if b == 0:
        va = '(eaB0 tape)'
    else:
        A = f'A{qb}'
        va = (f'(fun i k ↦ divInt (({A} : Nat) : ℤ) (alignNum (eaB{b} tape) HB{b} (maskM tape) '
              f'(RB{b} model key ext tape)\n        (K0B{b} model key) (K1B{b} model key) i k))')
    num = (f'cellNum {va}\n      {ds} (αB{b} ext) {es} (DrB{b} model key ext tape) (maskM tape)\n'
           f'      (XB{b} model key ext tape) (M0B{b} model key ext tape) (M1B{b} model key ext tape) i j')
    out.append(f'''theorem err{b + 1}_form (tape : SampleTape) :
    err{b + 1} model key ext tape = ∑ i, fun j ↦ divInt (({P} : Nat) : ℤ) ({num}) -
        ceB{b} ext i • rhsError{b} tape [2] i j :=
  rfl
''')
    return '\n'.join(out)


def lut_pieces(I):
    B, W = I.B, I.W
    m1 = I.m(1)
    return f'''/-! ## The readout LUT -/

noncomputable def ehL (tape : SampleTape) (i : Fin {W}) : Fin 2 → ErrorMatrix N 1 {m1} := helperError tape [3] i

noncomputable def DEL (tape : SampleTape) (i : Fin {W}) : ErrorMatrix N ({m1} + {m1}) {m1} :=
  liftMatrix (lutDigitsE (key{B} model key ext tape) (lutFam model key tape) i 1)

noncomputable def DDL (tape : SampleTape) (i : Fin {W}) : ErrorMatrix N ({m1} + {m1}) {m1} :=
  liftMatrix (lutDigitsD (lutFam model key tape) i 1)

def σL (i : Fin {W}) : Nat := (sigma i).toNat

noncomputable def sL (tape : SampleTape) (i : Fin {W}) : ErrorPoly N :=
  substitution (sigma i).toNat (sigma_odd i) (maskM tape)

noncomputable def KM0L (tape : SampleTape) (i : Fin {W}) : ErrorMatrix N 1 {m1} :=
  downResidue P2 (lutKeyM (key{B} model key ext tape) (lutFam model key tape) i 0 1)

noncomputable def KM1L (tape : SampleTape) (i : Fin {W}) : ErrorMatrix N 1 {m1} :=
  downResidue P2 (lutKeyM (key{B} model key ext tape) (lutFam model key tape) i 1 1)

noncomputable def RL (tape : SampleTape) : ErrorMatrix N 1 {m1} := downResidue P2 (lutRow model key ext tape 1)

noncomputable def K0L (tape : SampleTape) : ErrorMatrix N 1 {m1} :=
  downResidue P2 (lutKsum model key ext tape 0 1)

noncomputable def K1L (tape : SampleTape) : ErrorMatrix N 1 {m1} :=
  downResidue P2 (lutKsum model key ext tape 1 1)

noncomputable def aL (i : Fin {W}) : ErrorPoly N := substitution (sigma i).toNat (sigma_odd i) (muOf ext {B})

theorem lutNumer_form (tape : SampleTape) :
    lutNumer model key ext tape 1 = ∑ i, lutBranch (ehL tape) (DEL model key ext tape) σL sigma_odd
      (err{B} model key ext tape) (DDL model key tape) (sL tape) (KM0L model key ext tape)
      (KM1L model key ext tape) i - RL model key ext tape + maskM tape • K0L model key ext tape +
        K1L model key ext tape := by
  unfold lutNumer lutErr
  rw [Finset.sum_apply]
  simp only [lutBranchErr, lutBranch, add_assoc]
  rfl

theorem finalErr_form (tape : SampleTape) :
    finalErr model key ext tape 1 = (divInt ((P2 : Nat) : ℤ) (lutNumer model key ext tape 1) -
      ∑ i, aL ext i • switchError tape [2] i 1) * finalDigits :=
  rfl
'''


def weights(I):
    B, L = I.B, I.L
    m1 = I.m(1)
    out = [f'''/-! ## Weights -/

/-- The weight of the decoded coefficient `0`. -/
noncomputable def Wfin : Weight N 1 1 := fun _ _ k ↦ if k.val = 0 then 1 else 0

/-- The weight on the LUT error before the finalization. -/
noncomputable def WL : Weight N 1 {m1} := pullMul Wfin finalDigits

/-- The weight on the error entering the readout LUT. -/
noncomputable def V{B} (tape : SampleTape) (j' : Fin 2) : Weight N 1 {m1} :=
  (((P2 : Nat) : ℤ) : ℝ)⁻¹ • lutStateWeight hN WL (DDL model key tape) σL j'
''']
    for b in range(B - 1, -1, -1):
        qb = L - b
        ms = I.m(qb - 1)
        what = 'the state error of the encryption' if b == 0 else f'the error entering block `{b}`'
        out.append(f'''/-- The weight on {what}. -/
noncomputable def V{b} (tape : SampleTape) (j' : Fin 2) : Weight N 1 {ms} :=
  (((P{qb} : Nat) : ℤ) : ℝ)⁻¹ • stateWeight ({V(I, b + 1)}) (DrB{b} model key ext tape) (αB{b} ext) j'
''')
        ds = '(DsB0 model key)' if b == 0 else f'(DsB{b} model key ext tape)'
        out.append(f'''/-- The weights on the aligned members of block `{b}`, before the division by `P`. -/
noncomputable def U{b} (tape : SampleTape) : Fin {I.C} → Fin 2 → Weight N 1 {I.m(qb)} :=
  alignWeight ({V(I, b + 1)}) (DrB{b} model key ext tape) {ds}
''')
    return '\n'.join(out)


def nest(items):
    """Right-nested binary sum or `Sum.elim` of `items`."""
    if len(items) == 1:
        return items[0]
    return items, None


def sum_type(types):
    s = types[-1]
    for t in reversed(types[:-1]):
        s = f'{t} ⊕ ({s})' if ' ' in s and not s.startswith('(') else f'{t} ⊕ {s}'
    return s


def sum_elim(fs):
    s = fs[-1]
    for f in reversed(fs[:-1]):
        s = f'Sum.elim {f} ({s})' if not s.startswith('(') else f'Sum.elim {f} {s}'
    return s


def site_lists(I):
    """The site types and the coefficient, value and residue bound of every row and key site."""
    B, L, C = I.B, I.L, I.C
    m1 = I.m(1)
    xs = [f'(Fin {C} × Fin 2 × Fin {I.m(L - b - 2)} × Fin N)' for b in range(B)]
    as_ = [f'(Fin {C} × Fin 2 × Fin {I.m(L - b)} × Fin N)' for b in range(1, B)]
    rtypes = xs + as_ + [f'(Fin {m1} × Fin N)']
    ktypes = as_
    pr = lambda q: f'(((P{q} : Nat) : ℤ) : ℝ)'
    ar = lambda q: f'(((A{q} : Nat) : ℤ) : ℝ)'
    gR = ([f'(subCoef (fun _ j ↦ {V(I, b + 1)} j) {pr(L - b)})' for b in range(B)] +
          [f'(subCoef (U{b} model key ext tape) ({pr(L - b)} * {ar(L - b)}))' for b in range(1, B)] +
          [f'(lutCoef WL {pr(2)})'])
    rhoR = ([f'(siteVal (XB{b} model key ext tape))' for b in range(B)] +
            [f'(siteVal (RB{b} model key ext tape))' for b in range(1, B)] +
            ['(lutVal (RL model key ext tape))'])
    hR = ([f'(fun _ ↦ P{L - b} / 2)' for b in range(B)] + [f'(fun _ ↦ A{L - b} / 2)' for b in range(1, B)] +
          ['(fun _ ↦ P2 / 2)'])
    gK = [f'(maskCoef (U{b} model key ext tape) (maskM tape) ({pr(L - b)} * {ar(L - b)}))' for b in range(1, B)]
    rhoK = [f'(siteVal (K0B{b} model key))' for b in range(1, B)]
    hK = [f'(fun _ ↦ A{L - b} / 2)' for b in range(1, B)]
    return rtypes, ktypes, gR, rhoR, hR, gK, rhoK, hK


def site_sum(g, rho):
    return f'∑ x, {g} x * ({rho} x : ℝ)'


def nest_sum(xs):
    return xs[0] if len(xs) == 1 else f'({xs[0]} +\n      {nest_sum(xs[1:])})'


def sites(I):
    rtypes, ktypes, gR, rhoR, hR, gK, rhoK, hK = site_lists(I)
    rows = nest_sum([site_sum(g, r) for g, r in zip(gR, rhoR)])
    keys = nest_sum([site_sum(g, r) for g, r in zip(gK, rhoK)])
    return f'''/-! ## Sites -/

/-- The row residue sites: the private multiplications of every block, the alignments of every
block after the first, and the LUT finish. -/
abbrev RowSite : Type := {sum_type(rtypes)}

/-- The mask-weighted key residue sites of the alignments. -/
abbrev KeySite : Type := {sum_type(ktypes)}

/-- The H2 sites. -/
abbrev Site : Type := RowSite ⊕ KeySite

-- The site sums nest deeply; their instances are synthesized once here.
set_option synthInstance.maxSize 100000 in
set_option synthInstance.maxHeartbeats 2000000 in
instance : Fintype RowSite := inferInstance

set_option synthInstance.maxSize 100000 in
set_option synthInstance.maxHeartbeats 2000000 in
instance : Fintype KeySite := inferInstance

instance : Fintype Site := inferInstance

noncomputable def gRow (tape : SampleTape) : RowSite → ℝ :=
  {sum_elim(gR)}

noncomputable def rhoRow (tape : SampleTape) : RowSite → ℤ :=
  {sum_elim(rhoR)}

def hRow : RowSite → ℕ :=
  {sum_elim(hR)}

noncomputable def gKey (tape : SampleTape) : KeySite → ℝ :=
  {sum_elim(gK)}

noncomputable def rhoKey (tape : SampleTape) : KeySite → ℤ :=
  {sum_elim(rhoK)}

def hKey : KeySite → ℕ :=
  {sum_elim(hK)}

/-- The coefficient of a site. -/
noncomputable def siteG (j : Site) (tape : SampleTape) : ℝ :=
  Sum.elim (gRow model key ext tape) (gKey model key ext tape) j

/-- The value of a site. -/
noncomputable def siteRho (j : Site) (tape : SampleTape) : ℤ :=
  Sum.elim (rhoRow model key ext tape) (rhoKey model key tape) j

/-- The residue bound of a site. -/
def siteH (j : Site) : ℕ := Sum.elim hRow hKey j

/-! The site functions unfold by rewrites with proofs, never by the kernel's definitional
unfolding (see `elim_inl_eq`). -/

theorem gRow_eq (tape : SampleTape) : gRow model key ext tape =
    {sum_elim(gR)} := rfl.trans rfl

theorem rhoRow_eq (tape : SampleTape) : rhoRow model key ext tape =
    {sum_elim(rhoR)} := rfl.trans rfl

theorem gKey_eq (tape : SampleTape) : gKey model key ext tape =
    {sum_elim(gK)} := rfl.trans rfl

theorem rhoKey_eq (tape : SampleTape) : rhoKey model key tape =
    {sum_elim(rhoK)} := rfl.trans rfl

theorem siteG_inl (r : RowSite) (tape : SampleTape) :
    siteG model key ext (.inl r) tape = gRow model key ext tape r :=
  (Sum.elim_inl _ _ r).trans rfl

theorem siteG_inr (k : KeySite) (tape : SampleTape) :
    siteG model key ext (.inr k) tape = gKey model key ext tape k :=
  (Sum.elim_inr _ _ k).trans rfl

theorem siteRho_inl (r : RowSite) (tape : SampleTape) :
    siteRho model key ext (.inl r) tape = rhoRow model key ext tape r :=
  (Sum.elim_inl _ _ r).trans rfl

theorem siteRho_inr (k : KeySite) (tape : SampleTape) :
    siteRho model key ext (.inr k) tape = rhoKey model key tape k :=
  (Sum.elim_inr _ _ k).trans rfl

/-- The sum over the sites, one sum per site family. -/
theorem sum_sites (tape : SampleTape) :
    ∑ j, siteG model key ext j tape * (siteRho model key ext j tape : ℝ) =
      {rows} +
      {keys} := by
  simp only [Fintype.sum_sum_type, siteG_inl, siteG_inr, siteRho_inl, siteRho_inr, gRow_eq, rhoRow_eq,
    gKey_eq, rhoKey_eq, elim_inl_eq, elim_inr_eq]
'''


def steps(I):
    B, L, C = I.B, I.L, I.C
    out = ['/-! ## Decomposition -/\n']
    divs = ' ∧ '.join(f'div{b} model key ext tape' for b in range(B)) + ' ∧ divLut model key ext tape'
    out.append(f'''/-- Every ModDown numerator of the model is divisible by its dropped factor. -/
def Divisible (tape : SampleTape) : Prop :=
  {divs}

/-- The decoded coefficient of the model. -/
noncomputable def estar (tape : SampleTape) : ℝ := pairing Wfin (finalErr model key ext tape 1)
''')
    pr = lambda q: f'(((P{q} : Nat) : ℤ) : ℝ)'
    ar = lambda q: f'(((A{q} : Nat) : ℤ) : ℝ)'
    for b in range(B):
        qb = L - b
        P, Pr = f'(({"P" + str(qb)} : Nat) : ℤ)', pr(qb)
        es = '(stateError tape [3])' if b == 0 else f'(err{b} model key ext tape)'
        if b == 0:
            out.append(f'''/-- The remainder of the members of block `0`. -/
noncomputable def memberRem0 (tape : SampleTape) : ℝ :=
  (∑ i, ∑ k, pairing (U0 model key ext tape i k) (eaB0 tape i k)) / {Pr}

theorem step0 (tape : SampleTape) (hd : div0 model key ext tape) :
    pairing2 (V1 model key ext tape) (err1 model key ext tape) =
      pairing2 (V0 model key ext tape) {es} +
        ∑ x, subCoef (fun _ j ↦ V1 model key ext tape j) {Pr} x * (siteVal (XB0 model key ext tape) x : ℝ) +
        memberRem0 model key ext tape +
        cellRem (V1 model key ext tape) {Pr} (maskM tape) (M0B0 model key ext tape)
          (M1B0 model key ext tape) (ceB0 ext) (rhsError0 tape [2]) := by
  rw [err1_form]
  exact pairing2_firstBlock hN _ (by norm_num) _ _ _ _ _ _ _ _ _ hd _ _
''')
        else:
            A, Ar = f'((A{qb} : Nat) : ℤ)', ar(qb)
            out.append(f'''theorem step{b} (tape : SampleTape) (hd : div{b} model key ext tape) :
    pairing2 ({V(I, b + 1)}) (err{b + 1} model key ext tape) =
      pairing2 (V{b} model key ext tape) {es} +
        ∑ x, subCoef (fun _ j ↦ {V(I, b + 1)} j) {Pr} x * (siteVal (XB{b} model key ext tape) x : ℝ) +
        ∑ x, subCoef (U{b} model key ext tape) ({Pr} * {Ar}) x * (siteVal (RB{b} model key ext tape) x : ℝ) +
        ∑ x, maskCoef (U{b} model key ext tape) (maskM tape) ({Pr} * {Ar}) x *
          (siteVal (K0B{b} model key) x : ℝ) +
        alignRem (U{b} model key ext tape) ({Pr} * {Ar}) HB{b} (eaB{b} tape) (K1B{b} model key) +
        cellRem ({V(I, b + 1)}) {Pr} (maskM tape) (M0B{b} model key ext tape)
          (M1B{b} model key ext tape) (ceB{b} ext) (rhsError{b} tape [2]) := by
  rw [err{b + 1}_form]
  exact pairing2_alignedBlock hN _ (by norm_num) (by norm_num) _ _ _ _ _ _ _ _ _ _ _ _ _ hd.1 hd.2 _ _
''')
    out.append(f'''/-- The remainder of the readout LUT. -/
noncomputable def lutRemM (tape : SampleTape) : ℝ :=
  lutRem WL {pr(2)} (ehL tape) (DEL model key ext tape) (sL tape) (KM0L model key ext tape)
    (KM1L model key ext tape) (maskM tape) (K0L model key ext tape) (K1L model key ext tape) (aL ext)
    (fun i ↦ switchError tape [2] i 1)

theorem stepLut (tape : SampleTape) (hd : divLut model key ext tape) :
    estar model key ext tape = pairing2 (V{B} model key tape) (err{B} model key ext tape) +
      ∑ x, lutCoef WL {pr(2)} x * (lutVal (RL model key ext tape) x : ℝ) + lutRemM model key ext tape := by
  have hdiv := hd 1
  rw [lutNumer_form] at hdiv
  rw [estar, finalErr_form, lutNumer_form]
  exact pairing_lutFinal_sites hN Wfin finalDigits (by norm_num) _ _ _ _ _ _ _ _ _ _ _ _ _ hdiv _ _
''')
    rems = ['lutRemM model key ext tape'] + [
        f'alignRem (U{b} model key ext tape) ({pr(L - b)} * {ar(L - b)}) HB{b} (eaB{b} tape) (K1B{b} model key) +\n    '
        f'cellRem ({V(I, b + 1)}) {pr(L - b)} (maskM tape) (M0B{b} model key ext tape)\n      '
        f'(M1B{b} model key ext tape) (ceB{b} ext) (rhsError{b} tape [2])' for b in range(B - 1, 0, -1)] + [
        'memberRem0 model key ext tape',
        f'cellRem (V1 model key ext tape) {pr(L)} (maskM tape) (M0B0 model key ext tape)\n      '
        f'(M1B0 model key ext tape) (ceB0 ext) (rhsError0 tape [2])',
        'pairing2 (V0 model key ext tape) (stateError tape [3])']
    # The rearrangement of the decomposition over plain variables: the step `p`, member `m`, cell `c`,
    # alignment `r`, row `x`, alignment row `a` and key `k` terms, and the LUT sites `sl` and rest `rl`.
    vs = (['p', 'm', 'sl', 'rl'] + [f'x{b}' for b in range(B)] + [f'c{b}' for b in range(B)] +
          [f'{v}{b}' for v in 'akr' for b in range(1, B)])
    lhs = 'p + x0 + m + c0'
    for b in range(1, B):
        lhs = f'({lhs}) + x{b} + a{b} + k{b} + r{b} + c{b}'
    lhs = f'({lhs}) + sl + rl'
    rem = ' + '.join(['rl'] + [f'r{b} + c{b}' for b in range(B - 1, 0, -1)] + ['m', 'c0', 'p'])
    nest = lambda xs: xs[0] if len(xs) == 1 else f'({xs[0]} + {nest(xs[1:])})'
    rows = nest([f'x{b}' for b in range(B)] + [f'a{b}' for b in range(1, B)] + ['sl'])
    keys = nest([f'k{b}' for b in range(1, B)])
    out.append(f'''/-- The rearrangement of the decomposition. -/
theorem decomp_linear {{{' '.join(vs)} : ℝ}} :
    {lhs} =
      {rem} + ({rows} + {keys}) := by
  abel
''')
    hds = ', '.join(f'hd{b}' for b in range(B))
    rws = ', '.join(f'step{b} model key ext tape hd{b}' for b in range(B - 1, -1, -1))
    out.append(f'''/-- The remainder of the decoded coefficient: every term that is not an H2 site. -/
noncomputable def remainder (tape : SampleTape) : ℝ :=
  {(' +' + chr(10) + '  ').join(rems)}

set_option maxRecDepth 1000000 in
/-- On a tape whose numerators are divisible, the decoded coefficient is the remainder plus the
sum over the H2 sites. -/
theorem decomposition (tape : SampleTape) (hd : Divisible model key ext tape) :
    estar model key ext tape = remainder model key ext tape +
      ∑ j, siteG model key ext j tape * (siteRho model key ext j tape : ℝ) := by
  obtain ⟨{hds}, hdL⟩ := hd
  rw [stepLut model key ext tape hdL, {rws}, sum_sites, remainder]
  exact decomp_linear
''')
    return '\n'.join(out)


def module(I):
    body = '\n'.join(block_pieces(I, b) for b in range(I.B))
    return f'''import LwrModelLut
import ExponentH2

/-!
# The model in the form of the H2 decomposition

Generated by `generator/gen_h2_model.py`. The error leaving every block and the finalized error of
the readout LUT, written with the generic cell, alignment and LUT numerators of
`ExponentCellStep`; the weights pulled back from the decoded coefficient; the H2 sites (the row
residues of the private multiplications, of the alignments and of the LUT finish, and the
mask-weighted key residues of the alignments) with their coefficients and values; and the
decomposition of the decoded coefficient into a remainder and the sum over the sites.
-/

namespace LwrProof

open Mxx.Primitives MxxRuntime MxxExponentLut

section H2Model

variable (model : HashModel) (key : ByteArray) (ext : GeneratedClaim.ExternalInputs)

{body}
{lut_pieces(I)}
{weights(I)}
{sites(I)}
{steps(I)}
end H2Model

end LwrProof
'''


def main():
    d = sys.argv[1]
    I = Info(d)
    text = module(I)
    path = os.path.join(d, 'LwrH2Model.lean')
    with open(path, 'w') as f:
        f.write(text)
    print('wrote', path)


if __name__ == '__main__':
    main()
