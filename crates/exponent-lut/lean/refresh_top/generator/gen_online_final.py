"""Write `Part_<c_j>_OnlineFinal.lean`: the online root's output is, residue by residue modulo `q₀`,
the refresh's value `y = β A'₀ + A'₁ - T + e'`, from the semantic forms of the combination's rows,
the hint, the preimages, the smudges and the noise bound.

Usage: `python3 gen_online_final.py <parameter directory>`.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from chain import chain_binders
from gen_part import parts_of
from segment import abbreviations
from chains import chain_of
from layouts import layouts

d = sys.argv[1]
L = layouts(d)
ab = abbreviations(open(f'{d}/TopSetup.lean').read())
COLS = [(531, 85, 86, 579, (580, 581, 583)), (586, 584, 585, 634, (635, 636, 637)),
        (640, 638, 639, 688, (689, 690, 691)), (694, 692, 693, 742, (743, 744, 745)),
        (748, 746, 747, 796, (797, 798, 799)), (802, 800, 801, 850, (851, 852, 853))]
for part in parts_of(d):
    C = chain_of(part)
    Qq, qc = ('RefreshTopProof.' + ab[str(C['Qq'])] if str(C['Qq']) in ab else C['Qq']), C['qc']
    # Instance `I`: `Qq` as the integer table where lemmas take an integer (`QqN` where a natural
    # number), the rings' layouts and the chain's identities from the certificates of `TopInst`.
    inst = part == 'I'
    QqN = Qq
    cp = C['cp']
    Mx = 'RefreshTopProof.' + C['Mx']
    if inst:
        Qq, QqN, cp, Mx = '(QqZ (i0 : Int))', '(Qq i0)', '(cpZ (i0 : Int))', C['Mx']
    QqP = f'(({Qq} : Int) : ExactPoly Q1 N)'
    ns = f'Stage_online_eval_{part}'
    _t = open(f'{d}/generated/{ns}.lean').read()
    CX = re.search(r'let w_356_0 [^\n]*:= MxxRuntime\.matrixMul (w_\d+_0) w_355_0', _t).group(1)
    atoms = ['h335_0', 'h339_0', 'h341_0', 'h342_0', 'h351_0', 'h354_0', 'h355_0',
             'h637_0', 'h691_0', 'h745_0', 'h799_0', 'h854_0', 'h855_0'] + [f'h{c[0]}_0' for c in COLS]
    extra = ['w_340_0', 'w_336_0', 'w_352_0', 'w_353_0', 'w_530_0', 'w_333_0', 'w_334_0', CX, 'w_529_0', 'w_84_0'] + \
        [f'w_{c[4][2]}_0' for c in COLS] + [f'w_{c[1]}_0' for c in COLS] + [f'w_{c[3]}_5' for c in COLS] + \
        [f'w_{c[2]}_0' for c in COLS] + ['w_350_0']
    binders = chain_binders(f'{d}/generated/{ns}.lean', ns, [], ab, atoms, extra)
    hyps = ['''    {sm : ErrorPoly N} {β : ExactPoly Q1 N} {x : ErrorPoly N}
    {Kg K₁ Kw : Fin 2 → ExactMatrix Q1 N 1 6} {Kx : Fin 2 → ExactMatrix Q1 N 1 4}
    {Bh : Fin 2 → ExactMatrix Q1 N 1 16} {e' : ExactMatrix Q1 N 1 6}
    {GQ1_6 : ExactMatrix Q1 N 1 6} (hGQ1_6 : gadgetMatrixRuns Backend.backend 32768 6 GQ1_6)
    {GM01_4 : ExactMatrix ''' + Mx + ''' N 1 4} {RQ1_4 : ExactMatrix Q1 N 1 4}
    (hGM01_4 : gadgetMatrixRuns Backend.backend 32768 4 GM01_4) (hRQ1_4 : centeredRebaseRuns GM01_4 RQ1_4)
    {egI e1I ewI : ErrorMatrix N 1 6} {exI : ErrorMatrix N 1 4} {ebI : ErrorMatrix N 1 16}
    {Beg Be1 Bew Bex Beb Bpre BM : Nat}
    (heg : CoeffBound egI Beg) (he1 : CoeffBound e1I Be1) (hew : CoeffBound ewI Bew)
    (hex : CoeffBound exI Bex) (heb : CoeffBound ebI Beb)
    (hg : w_333_0 = maskPart (reducePoly Q1 N sm) Kg - β • GQ1_6 + reduceMatrix Q1 N 1 6 egI)
    (hc₁ : w_334_0 = maskPart (reducePoly Q1 N sm) K₁ - GQ1_6 + reduceMatrix Q1 N 1 6 e1I)
    (hcx : ''' + CX + ''' = maskPart (reducePoly Q1 N sm) Kx - reducePoly Q1 N (((''' + C['p'] + ''' : Nat) : Int) * x) • RQ1_4 +
      reduceMatrix Q1 N 1 4 exI)
    (hfw : w_529_0 = maskPart (reducePoly Q1 N sm) Kw - ''' + QqP + ''' • e' + reduceMatrix Q1 N 1 6 ewI)
    (hb : w_84_0 = maskPart (reducePoly Q1 N sm) Bh + reduceMatrix Q1 N 1 16 ebI)''']
    for k, (sl, P, bK, SM, (a, b, o)) in enumerate(COLS):
        hyps.append(f'''    {{KI{k} : ErrorMatrix N 16 1}} (hK{k} : w_{P}_0 = reduceMatrix Q1 N 16 1 KI{k}) (hKb{k} : CoeffBound KI{k} Bpre)
    {{Ksm{k} : Fin 2 → ExactMatrix Q1 N 1 1}} {{V{k} M{k} : ErrorPoly N}} {{esm{k} : ErrorMatrix N 1 1}} {{Besm{k} : Nat}}
    (hpre{k} : ∀ i, Bh i * w_{P}_0 = (Kg i * topRows w_342_0 + K₁ i * bottomRows w_342_0 - Kx i * w_355_0 + Kw i) *
      unitCol (q := Q1) {k} + Ksm{k} i)
    (hsm{k} : w_{SM}_5 = maskPart (reducePoly Q1 N sm) Ksm{k} - reducePoly Q1 N V{k} • (1 : ExactMatrix Q1 N 1 1) +
      reduceMatrix Q1 N 1 1 esm{k}) (hesm{k} : CoeffBound esm{k} Besm{k})
    (hV{k} : reducePoly Q1 N V{k} - w_582_0 0 0 = reducePoly Q1 N M{k})
    (hM{k} : CoeffBound (fun _ _ ↦ M{k} : ErrorMatrix N 1 1) BM)
    (hnoise{k} : 2 * (BM + 16 * N * Beb * Bpre + (6 * N * Beg * 16384 + 6 * N * Be1 * 16384 +
      4 * N * Bex * 16384 + Bew) + Besm{k}) < {QqN})''')
    proof = (['''  obtain ⟨L, hL, hLx, hLb, hLd⟩ := layoutR i0 hi0
  obtain ⟨LX, hLX, hLXx, -, -⟩ := layoutMx i0 hi0'''] if inst else []) + ['''  obtain ⟨DI, hDI, hDIb⟩ := h342_0.2.2.2
  obtain ⟨YI, hYI, hYIb⟩ := h355_0.2.2.2''']
    cols = []
    for k, (sl, P, bK, SM, (a, b, o)) in enumerate(COLS):
        if inst:
            layout_args = (f"L hL hLx (by inst_num) (by rw [hLb]; norm_num) (by rw [hLd]; norm_num) (by rw [hLd])\n"
                           f"    (p := {C['p']}) (Mx := {Mx}) (R_eq_pMx i0 hi0) (by inst_num) LX hLX hLXx (action_cert i0 hi0)")
        else:
            layout_args = (f"Backend.{L[C['R']]} {C['hlayoutR']} rfl\n    (by norm_num) rfl rfl rfl (p := {C['p']}) "
                           f"(Mx := RefreshTopProof.{C['Mx']}) (by norm_num) (by norm_num) Backend.{L[C['Mx']]} {C['hlayoutX']} rfl (by norm_num)")
        proof.append(f'''  have hcol{k} := column_valueR (Qq := {Qq}) (c' := {cp}) (k := actionConst' {cp} {C['kdiv']}) {layout_args}
    (G := GQ1_6) (H := w_582_0) h335_0 (z := w_340_0) (A' := w_339_0) (by rw [e_340_0, e_336_0]) h341_0 h342_0 hGQ1_6
    h351_0 (by rw [e_353_0, e_352_0]) h354_0 h355_0 hGM01_4 hRQ1_4 (comb := w_530_0)
    (by rw [e_530_0, e_357_0, e_343_0, e_356_0]) (by decide : (({k} : Int)).toNat < 6) h{sl}_0 hg hc₁ hcx hfw hb
    (Kpre := w_{P}_0) hpre{k} hsm{k} (out := w_{o}_0) (by rw [e_{o}_0, e_{b}_0, e_{a}_0, e_{bK}_0]; try rfl)
  have hSH{k} : reducePoly Q1 N V{k} • (1 : ExactMatrix Q1 N 1 1) - w_582_0 =
      reduceMatrix Q1 N 1 1 (fun _ _ ↦ M{k}) := by
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _
    simp only [Matrix.sub_apply, Matrix.smul_apply, Matrix.one_apply_eq, smul_eq_mul, mul_one, reduceMatrix]
    exact hV{k}
  rw [hSH{k}, hK{k}, hDI, hYI] at hcol{k}
  obtain ⟨ν{k}, hν{k}, hc{k}⟩ := column_noiseQ (Qq := {Qq}) (y := β • rowOf w_339_0 0 + rowOf w_339_0 1 - actionValueC (actionConst' {cp} {C['kdiv']}) x w_350_0 + e')
    {k} hM{k} heb hKb{k} heg he1 hew (X0I := topRows DI) (X1I := bottomRows DI)
    (fun r c i ↦ hDIb _ c i) (fun r c i ↦ hDIb _ c i) hex hYIb hesm{k} (out := w_{o}_0) (by rw [hcol{k}]; try rfl)''')
        cols.append(f'w_{o}_0')
    proof.append(f'''  have hz := concat_six h637_0 h691_0 h745_0 h799_0 h854_0
  intro k i
  refine switch_residue (Qq := {QqN}) (by norm_num) (by norm_num) (by norm_num) h855_0
    (y := β • rowOf w_339_0 0 + rowOf w_339_0 1 - actionValueC (actionConst' {cp} {C['kdiv']}) x w_350_0 + e')
    (ν := ![ν0 0 0, ν1 0 0, ν2 0 0, ν3 0 0, ν4 0 0, ν5 0 0]) (fun k ↦ ?_) (fun k j ↦ ?_) k i
  · {'simp only [Qq_cast]' + chr(10) + '    ' if inst else ''}rw [hz k]
    fin_cases k
    · exact hc0
    · exact hc1
    · exact hc2
    · exact hc3
    · exact hc4
    · exact hc5
  · fin_cases k
    · have := hν0 0 0 j
      show 2 * |(ν0 0 0).coeff j| < (({QqN} : Nat) : Int)
      rw [Int.abs_eq_natAbs]; omega
    · have := hν1 0 0 j
      show 2 * |(ν1 0 0).coeff j| < (({QqN} : Nat) : Int)
      rw [Int.abs_eq_natAbs]; omega
    · have := hν2 0 0 j
      show 2 * |(ν2 0 0).coeff j| < (({QqN} : Nat) : Int)
      rw [Int.abs_eq_natAbs]; omega
    · have := hν3 0 0 j
      show 2 * |(ν3 0 0).coeff j| < (({QqN} : Nat) : Int)
      rw [Int.abs_eq_natAbs]; omega
    · have := hν4 0 0 j
      show 2 * |(ν4 0 0).coeff j| < (({QqN} : Nat) : Int)
      rw [Int.abs_eq_natAbs]; omega
    · have := hν5 0 0 j
      show 2 * |(ν5 0 0).coeff j| < (({QqN} : Nat) : Int)
      rw [Int.abs_eq_natAbs]; omega''')
    body = f'''import TopColumn
import TopTarget
import TopResidue
import TopChain
import {ns}

/-! The online root's output of part `{part}` (written by `generator/gen_online_final.py`). -/

namespace RefreshTopProof.Part_{part}

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof Mxx.ExponentLut.Refresh

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The online root's output: modulo `q₀`, the refresh's value `y`. -/
theorem onlineFinal {{model : HashModel}} {{params : {ns}.Params}}
''' + '\n'.join('    ' + b for b in binders) + '\n' + '\n'.join(hyps) + ''' :
    ∀ (k : Fin 6) (i : Fin N), (w_855_0 0 k).coeff i =
      ((((β • rowOf w_339_0 0 + rowOf w_339_0 1 - actionValueC (actionConst' ''' + str(cp) + ' ' + C['kdiv'] + ''') x w_350_0 + e' : ExactMatrix Q1 N 1 6) 0 k).coeff i).val : ZMod ''' + qc + ''') := by
''' + '\n'.join(proof) + f'''

end RefreshTopProof.Part_{part}
'''
    open(f'{d}/Part_{part}_OnlineFinal.lean', 'w').write(body)
    break
