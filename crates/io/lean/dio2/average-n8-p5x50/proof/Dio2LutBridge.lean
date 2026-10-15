import Dio2Lut
import Dio2Comp
import GenKey
import GenRelLut

/-!
# The bundle bridge of a dio2 LUT loop step

`LutRoundStep.step` destructures the computation bundle `LutRoundStep.Comp` (variant 0) of one
iteration `i` of a LUT loop and applies `lut_branch_flat`: the accumulated relation gains branch
`i` (automorphism `σ_i = sigmas[i]`, whose helper is the root row of the helper tagged `σ_i`), and
the carried switching ciphertexts move to branch `i + 1`.
-/

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof.Gen

variable {n : Nat}

/-- A dynamic family read at a natural index is the member at that index. -/
theorem familyGetDynamic_nat {α : Type} {W : Nat} {f : Fin W → α} {i : Nat} {x : α}
    (h : familyGetDynamic f (Int.ofNat i) x) (hi : i < W) : x = f ⟨i, hi⟩ := by
  obtain ⟨pos, hpos, rfl⟩ := h
  congr 1
  exact Fin.ext (by simp only [Int.ofNat_eq_natCast] at hpos; simp only []; omega)

/-- The successor form of `familyGetDynamic_nat`. -/
theorem familyGetDynamic_succ {α : Type} {W : Nat} {f : Fin W → α} {i : Nat} {x : α}
    (h : familyGetDynamic f (Int.ofNat i + 1) x) (hi : i + 1 < W) : x = f ⟨i + 1, hi⟩ := by
  obtain ⟨pos, hpos, rfl⟩ := h
  congr 1
  exact Fin.ext (by simp only [Int.ofNat_eq_natCast] at hpos; simp only []; omega)

set_option maxHeartbeats 16000000 in
/-- **One LUT loop step** (variant 0): the accumulated relation `(keys, rows)` of `Tc` gains
branch `i`, and the carried switching ciphertexts become member `i + 1` of their families. -/
theorem LutRoundStep.step (hn : 0 < n) {log : Nat} (hpow : n = 2 ^ log)
    {q m m' bound W : Nat} (hq1 : 1 < q)
    {backend : BackendContext} {hashModel : HashModel} {base digits : Int}
    (layout : RegularLayout q) (hlayout : backend.regularLayout q n = some layout)
    (hexact : layout.droppedModuli = 0)
    {lo0 hi0 lo1 hi1 l2 h2 l3 h3 l4 h4 l5 h5 : Int} {pre : Blob} {sigmas : List Int}
    {seed : ByteArray} {i : Nat}
    {r0 r1 k00 k01 k10 k11 b00 b01 b10 b11 : ExactMatrix q n 1 m'}
    {hf0 hf1 : Fin W → ExactMatrix q n 1 m} {c0 a00 a10 c1 a01 a11 : ExactMatrix q n 1 m}
    {sw00 sw01 sw10 sw11 : Fin W → ExactMatrix q n 1 m'}
    {outputs : ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' ×
      ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' ×
      ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × Unit}
    (h : Dio2Proof.Comp.LutRoundStep.Comp q n m' m' m' m' m' m' m' m' m' m' q q W m W m m m W
      q n m' W q n m' W q n m' W q n m' lo0 hi0 pre sigmas pre sigmas lo1 hi1 pre sigmas pre
      sigmas sigmas sigmas sigmas sigmas sigmas sigmas base digits bound m' m' base digits bound
      m' m' base digits bound m' m' base digits bound m' m' m' m' m' m' l2 h2 l3 h3 l4 h4 l5 h5
      backend hashModel i
      (r0, r1, k00, k01, k10, k11, b00, b01, b10, b11, hf0, seed, hf1, c0, a00, a10, c1, a01, a11,
        sw00, sw01, sw10, sw11, ()) outputs)
    (hi : i < W) (hodd : Odd (MxxIR.selectInt (Int.ofNat i) sigmas).toNat)
    {sm sp μ : ErrorPoly n} (hsm : polyNorm sm ≤ 1)
    {g' : ExactMatrix q n 1 m} (hg' : gadgetMatrixRuns backend base digits g')
    {g : ExactMatrix q n 1 m} (hg : centeredRebaseRuns g' g) {B Bh Bs : Nat}
    (henc : Rel sm ![![a00, a01], ![a10, a11]]
      (fun j ↦ reducePoly q n (((1 : Nat) : Int) * μ * unitPair sp j) • g) ![c0, c1] B)
    (hhelp : Rel sm (keyPairG hashModel seed q m pre [.integer (MxxIR.selectInt (Int.ofNat i) sigmas)])
      (fun j ↦ reducePoly q n (((1 : Nat) : Int) *
        unitPair (substitution (MxxIR.selectInt (Int.ofNat i) sigmas).toNat hodd sm) j) • g)
      ![hf0 ⟨i, hi⟩, hf1 ⟨i, hi⟩] Bh)
    {Tc : Fin 2 → ExactMatrix q n 1 m'}
    (hcur : Rel sm ![![k00, k01], ![k10, k11]] Tc ![r0, r1] Bs) :
    Rel sm ![![outputs.2.2.1, outputs.2.2.2.1], ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1]]
        (fun j ↦ Tc j + reducePoly q n (substitution (MxxIR.selectInt (Int.ofNat i) sigmas).toNat
          hodd μ) • (reducePoly q n (substitution (MxxIR.selectInt (Int.ofNat i) sigmas).toNat
            hodd sp) • ![b00, b01] j + ![b10, b11] j))
        ![outputs.1, outputs.2.1] (Bs + ((m + m) * n * Bh * bound + (m + m) * n * B * bound)) ∧
      ∀ hi1 : i + 1 < W, outputs.2.2.2.2.2.2.1 = sw00 ⟨i + 1, hi1⟩ ∧
        outputs.2.2.2.2.2.2.2.1 = sw01 ⟨i + 1, hi1⟩ ∧
        outputs.2.2.2.2.2.2.2.2.1 = sw10 ⟨i + 1, hi1⟩ ∧
        outputs.2.2.2.2.2.2.2.2.2.1 = sw11 ⟨i + 1, hi1⟩ := by
  obtain ⟨v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
    v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37,
    v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49,
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20,
    c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39,
    c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51, c52, c53, c54, c55, c56, c57, c58,
    c59, c60, c61, c62, h⟩ := h
  subst h
  have e0 := familyGetDynamic_nat c3 hi
  have e5 := familyGetDynamic_nat c10 hi
  subst e0 e5
  -- The helper's keys are the root keys of the helper tagged `σ_i`.
  have hkeys : ![![v1, v6], ![v2, v7]] = keyPairG hashModel seed q m pre
      [.integer (MxxIR.selectInt (Int.ofNat i) sigmas)] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact keyPairG_of_hash (pre := [.integer (MxxIR.selectInt (Int.ofNat i) sigmas)]) 0 0
        (by norm_num) c4
    · exact keyPairG_of_hash (pre := [.integer (MxxIR.selectInt (Int.ofNat i) sigmas)]) 0 1
        (by norm_num) c11
    · exact keyPairG_of_hash (pre := [.integer (MxxIR.selectInt (Int.ofNat i) sigmas)]) 1 0
        (by norm_num) c5
    · exact keyPairG_of_hash (pre := [.integer (MxxIR.selectInt (Int.ofNat i) sigmas)]) 1 1
        (by norm_num) c12
  rw [← hkeys] at hhelp
  have hb := lut_branch_flat hn hpow hq1 layout hlayout hexact hg' hg hsm henc
    (rc0 := v11) (rA00 := v12) (rA10 := v13) (rc1 := v16) (rA01 := v17) (rA11 := v18)
    c16 c17 c18 c21 c22 c23 c19 c20 c24 c25 c26
    (bsw := ![![b00, b01], ![b10, b11]]) (Xb := ![v22, v32])
    (fun j ↦ by fin_cases j; exacts [c27, c37])
    (D' := ![v23, v33]) (fun j ↦ by fin_cases j; exacts [c28, c38])
    (D := ![v24, v34]) (fun j ↦ by fin_cases j; exacts [c29, c39])
    (M := ![![v25, v35], ![v26, v36]]) (fun j ↦ by fin_cases j; exacts [c30, c40])
    (fun j ↦ by fin_cases j; exacts [c31, c41])
    (Z := ![v27, v37]) (fun j ↦ by fin_cases j; exacts [c32, c42])
    (E' := ![v28, v38]) (fun j ↦ by fin_cases j; exacts [c33, c43])
    (E := ![v29, v39]) (fun j ↦ by fin_cases j; exacts [c34, c44])
    hodd hhelp c6 c7 c13 c14 c15
    (Y := ![v30, v40]) (X := ![v31, v41]) (K := ![![v42, v43], ![v44, v45]])
    (fun j ↦ by fin_cases j; exacts [c35, c45])
    (fun j ↦ by fin_cases j; exacts [c36, c46])
    (fun j ↦ by fin_cases j; exacts [c47, c48])
    (fun j ↦ by fin_cases j; exacts [c49, c50])
  refine ⟨?_, fun hi1 ↦ ⟨familyGetDynamic_succ c53 hi1, familyGetDynamic_succ c56 hi1,
    familyGetDynamic_succ c59 hi1, familyGetDynamic_succ c62 hi1⟩⟩
  have hsum := hcur.add hb
  have hK : ![![k00, k01], ![k10, k11]] + ![![v42, v43], ![v44, v45]] =
      ![![matrixAdd k00 v42, matrixAdd k01 v43], ![matrixAdd k10 v44, matrixAdd k11 v45]] := by
    funext k j
    fin_cases k <;> fin_cases j <;> rfl
  have hR : ![r0, r1] + (fun j ↦ ![v30, v40] j + ![v31, v41] j) =
      ![matrixAdd r0 (matrixAdd v30 v31), matrixAdd r1 (matrixAdd v40 v41)] := by
    funext j
    fin_cases j <;> rfl
  rw [hK, hR] at hsum
  exact hsum


/-- The automorphism of branch `i` of a LUT whose table of automorphisms is `sigmas`. -/
def sigmaOf {W : Nat} (sigmas : List Int) (i : Fin W) : Int := MxxIR.selectInt (Int.ofNat i.val) sigmas

set_option maxHeartbeats 64000000 in
/-- **A LUT member at one modulus** (`LutMemberFlat.Comp`): the generated loop over the first
`L = W - 1` branches (each step `LutRoundStep.step`), the last branch and the finish. The input
`(c, A)` encodes `μ` (scale one) against `g`, branch `i`'s helper is the root row of the helper
tagged `σ_i` (`hhelp`), and the switching ciphertexts encode the LUT coefficients `C_i` with
`Σ_i σ_i(μ) C_i = ν` (`hswitch`, `hwalsh`): the member encodes `Δ' ν` against `gq`. -/
theorem LutMemberFlat.rel (hn : 0 < n) {log : Nat} (hpow : n = 2 ^ log)
    {q m m' bound W L : Nat} (hL : L + 1 = W) (hq1 : 1 < q)
    {backend : BackendContext} {hashModel : HashModel} {base digits : Int}
    (layout : RegularLayout q) (hlayout : backend.regularLayout q n = some layout)
    (hexact : layout.droppedModuli = 0)
    {pre : Blob} {sigmas : List Int} {seed : ByteArray}
    {hf0 hf1 : Fin W → ExactMatrix q n 1 m} {c0 a00 a10 c1 a01 a11 : ExactMatrix q n 1 m}
    {sw00 sw01 sw10 sw11 : Fin W → ExactMatrix q n 1 m'}
    {lo0 hi0 lo1 hi1 l2 h2 l3 h3 l4 h4 l5 h5 : Int}
    {child : Nat → ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' ×
      ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' ×
      ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' ×
      (Fin W → ExactMatrix q n 1 m) × ByteArray × (Fin W → ExactMatrix q n 1 m) ×
      ExactMatrix q n 1 m × ExactMatrix q n 1 m × ExactMatrix q n 1 m × ExactMatrix q n 1 m ×
      ExactMatrix q n 1 m × ExactMatrix q n 1 m × (Fin W → ExactMatrix q n 1 m') ×
      (Fin W → ExactMatrix q n 1 m') × (Fin W → ExactMatrix q n 1 m') ×
      (Fin W → ExactMatrix q n 1 m') × Unit →
      ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' ×
      ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' ×
      ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × Unit → Prop}
    (hchild : ∀ i a b, child i a b → Dio2Proof.Comp.LutRoundStep.Comp q n m' m' m' m' m' m' m' m'
      m' m' q q W m W m m m W q n m' W q n m' W q n m' W q n m' lo0 hi0 pre sigmas pre sigmas lo1
      hi1 pre sigmas pre sigmas sigmas sigmas sigmas sigmas sigmas sigmas base digits bound m' m'
      base digits bound m' m' base digits bound m' m' base digits bound m' m' m' m' m' m' l2 h2 l3
      h3 l4 h4 l5 h5 backend hashModel i a b)
    {outputs : ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' ×
      ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × Unit}
    (h : Dio2Proof.Comp.LutMemberFlat.Comp q n m' q m' q m' W m W m m m W W W W m' m'
      0 0 0 0 (L : Int) (L : Int) pre (L : Int) sigmas pre (L : Int) sigmas (L : Int) pre (L : Int)
      sigmas pre (L : Int) sigmas (L : Int) sigmas (L : Int) sigmas (L : Int) sigmas (L : Int) sigmas
      (L : Int) sigmas (L : Int) sigmas base digits bound m' m' base digits bound m' base digits
      bound m' m' base digits bound m' m' m' m' m' m' m' backend hashModel child
      (sw00, sw01, sw10, sw11, hf0, seed, hf1, c0, a00, a10, c1, a01, a11, ()) outputs)
    (hodd : ∀ i : Fin W, Odd (sigmaOf sigmas i).toNat)
    {sm sp μ : ErrorPoly n} (hsm : polyNorm sm ≤ 1)
    {g' : ExactMatrix q n 1 m} (hg' : gadgetMatrixRuns backend base digits g')
    {g : ExactMatrix q n 1 m} (hg : centeredRebaseRuns g' g) {B Bh : Nat}
    (henc : Rel sm ![![a00, a01], ![a10, a11]]
      (fun j ↦ reducePoly q n (((1 : Nat) : Int) * μ * unitPair sp j) • g) ![c0, c1] B)
    (hhelp : ∀ i : Fin W, Rel sm (keyPairG hashModel seed q m pre [.integer (sigmaOf sigmas i)])
      (fun j ↦ reducePoly q n (((1 : Nat) : Int) *
        unitPair (substitution (sigmaOf sigmas i).toNat (hodd i) sm) j) • g) ![hf0 i, hf1 i] Bh)
    {Δ' ν : ErrorPoly n} {gq : ExactMatrix q n 1 m'} {C : Fin W → ExactPoly q n} {BB : Nat}
    (hswitch : ∀ i : Fin W, ∃ e : Fin 2 → ErrorMatrix n 1 m', (∀ j, CoeffBound (e j) BB) ∧ ∀ j,
      reducePoly q n (substitution (sigmaOf sigmas i).toNat (hodd i) sp) •
          ![![sw00, sw01], ![sw10, sw11]] 0 j i + ![![sw00, sw01], ![sw10, sw11]] 1 j i =
        (C i * reducePoly q n (Δ' * unitPair sp j)) • gq + reduceMatrix q n 1 m' (e j))
    (hμ : ∀ i, polyNorm (substitution (sigmaOf sigmas i).toNat (hodd i) μ) ≤ 1)
    (hwalsh : ∑ i, reducePoly q n (substitution (sigmaOf sigmas i).toNat (hodd i) μ) * C i =
      reducePoly q n ν) :
    Rel sm ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]]
      (fun j ↦ reducePoly q n (Δ' * ν * unitPair sp j) • gq)
      ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1]
      (L * ((m + m) * n * Bh * bound + (m + m) * n * B * bound) +
        ((m + m) * n * Bh * bound + (m + m) * n * B * bound) + W * (n * 1 * BB)) := by
  obtain ⟨v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
    v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37,
    v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55, v56,
    v57, v58, v59, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18,
    c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37,
    c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51, h⟩ := h
  subst h
  have hW : 0 < W := by omega
  let fam : Fin 2 → Fin 2 → Fin W → ExactMatrix q n 1 m' := ![![sw00, sw01], ![sw10, sw11]]
  -- The loop over the first `L` branches.
  have hloop := lut_loop_gen (n := n) (S := ExactMatrix q n 1 m' × ExactMatrix q n 1 m' ×
      ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' ×
      ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' ×
      Unit) (W := W) (sm := sm)
    (fun s ↦ ![![s.2.2.1, s.2.2.2.1], ![s.2.2.2.2.1, s.2.2.2.2.2.1]]) (fun s ↦ ![s.1, s.2.1])
    (fun s ↦ ![![s.2.2.2.2.2.2.1, s.2.2.2.2.2.2.2.1], ![s.2.2.2.2.2.2.2.2.1, s.2.2.2.2.2.2.2.2.2.1]])
    (lutBranchFlat (sigmaOf sigmas) hodd μ sp fam) fam
    ((m + m) * n * Bh * bound + (m + m) * n * B * bound) (by omega)
    (by
      have hk : (![![(0 : ExactMatrix q n 1 m'), 0], ![0, 0]] : Fin 2 → Fin 2 → _) = 0 := by
        funext k j; fin_cases k <;> fin_cases j <;> rfl
      have h0 : (![(0 : ExactMatrix q n 1 m'), 0] : Fin 2 → _) = 0 := by
        funext j; fin_cases j <;> rfl
      show Rel sm ![![(0 : ExactMatrix q n 1 m'), 0], ![0, 0]] 0 ![0, 0] 0
      rw [hk, h0]
      exact Rel.zero)
    (by
      funext k j
      fin_cases k <;> fin_cases j
      · exact familyGetDynamic_nat (i := 0) c1 hW
      · exact familyGetDynamic_nat (i := 0) c2 hW
      · exact familyGetDynamic_nat (i := 0) c3 hW
      · exact familyGetDynamic_nat (i := 0) c4 hW)
    (fun i hi cur nxt Tc Bs hbody hsw hcur ↦ by
      have hstep := LutRoundStep.step hn hpow hq1 layout hlayout hexact (hchild _ _ _ hbody)
        (by omega) (hodd ⟨i, by omega⟩) hsm hg' hg henc (hhelp ⟨i, by omega⟩) hcur
      obtain ⟨hrel, hnext⟩ := hstep
      have e00 : cur.2.2.2.2.2.2.1 = fam 0 0 ⟨i, by omega⟩ := by
        simpa using congrFun (congrFun hsw 0) 0
      have e01 : cur.2.2.2.2.2.2.2.1 = fam 0 1 ⟨i, by omega⟩ := by
        simpa using congrFun (congrFun hsw 0) 1
      have e10 : cur.2.2.2.2.2.2.2.2.1 = fam 1 0 ⟨i, by omega⟩ := by
        simpa using congrFun (congrFun hsw 1) 0
      have e11 : cur.2.2.2.2.2.2.2.2.2.1 = fam 1 1 ⟨i, by omega⟩ := by
        simpa using congrFun (congrFun hsw 1) 1
      refine ⟨hrel.target_eq ?_, ?_⟩
      · funext j
        simp only [lutBranchFlat, dif_pos (show i < W by omega), branchFlat, Pi.add_apply]
        fin_cases j
        · simp only [Fin.zero_eta, Fin.isValue, Matrix.cons_val_zero, e00, e10]; rfl
        · simp only [Fin.mk_one, Fin.isValue, Matrix.cons_val_one, Matrix.cons_val_zero, e01,
            e11]; rfl
      · obtain ⟨e0, e1, e2, e3⟩ := hnext hi
        funext k j
        fin_cases k <;> fin_cases j
        · exact e0
        · exact e1
        · exact e2
        · exact e3)
    c5
  obtain ⟨hrel, hsw⟩ := hloop
  simp only [Int.toNat_natCast] at hrel hsw
  have hs00 := congrFun (congrFun hsw 0) 0
  have hs01 := congrFun (congrFun hsw 0) 1
  have hs10 := congrFun (congrFun hsw 1) 0
  have hs11 := congrFun (congrFun hsw 1) 1
  simp only [Fin.isValue, Matrix.cons_val_zero, Matrix.cons_val_one] at hs00 hs01 hs10 hs11
  have e14 := familyGetDynamic_nat (i := L) c6 (by omega)
  have e19 := familyGetDynamic_nat (i := L) c11 (by omega)
  subst e14 e19
  have hkeys : ![![v15, v20], ![v16, v21]] = keyPairG hashModel seed q m pre
      [.integer (sigmaOf (W := W) sigmas ⟨L, by omega⟩)] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact keyPairG_of_hash (pre := [.integer (sigmaOf (W := W) sigmas ⟨L, by omega⟩)]) 0 0 (by norm_num) c7
    · exact keyPairG_of_hash (pre := [.integer (sigmaOf (W := W) sigmas ⟨L, by omega⟩)]) 0 1 (by norm_num) c12
    · exact keyPairG_of_hash (pre := [.integer (sigmaOf (W := W) sigmas ⟨L, by omega⟩)]) 1 0 (by norm_num) c8
    · exact keyPairG_of_hash (pre := [.integer (sigmaOf (W := W) sigmas ⟨L, by omega⟩)]) 1 1 (by norm_num) c13
  have hhelpL := hhelp ⟨L, by omega⟩
  rw [← hkeys] at hhelpL
  have hround := lut_round_flat hn hpow hL hq1 layout hlayout hexact (sigmaOf sigmas) hodd hsm hg'
    hg henc (fam := fam) hrel (h3 := ![hf0 ⟨L, by omega⟩, hf1 ⟨L, by omega⟩]) hhelpL
    (rc0 := v25) (rA00 := v26) (rA10 := v27) (rc1 := v30) (rA01 := v31) (rA11 := v32)
    c17 c18 c19 c22 c23 c24 c20 c21 c25 c26 c27
    (Xb := ![v36, v45]) (fun j ↦ by
      fin_cases j
      · show concatRows (fam 0 0 _) (fam 1 0 _) v36; rw [← hs00, ← hs10]; exact c28
      · show concatRows (fam 0 1 _) (fam 1 1 _) v45; rw [← hs01, ← hs11]; exact c37)
    (D' := ![v37, v46]) (fun j ↦ by fin_cases j; exacts [c29, c38])
    (D := ![v38, v47]) (fun j ↦ by fin_cases j; exacts [c30, c39])
    (M := ![![v39, v48], ![v40, v49]]) (fun j ↦ by fin_cases j; exacts [c31, c40])
    (fun j ↦ by fin_cases j; exacts [c32, c41])
    (Z := ![v41, v50]) (fun j ↦ by fin_cases j; exacts [c33, c42])
    (E' := ![v42, v51]) (fun j ↦ by fin_cases j; exacts [c34, c43])
    (E := ![v43, v52]) (fun j ↦ by fin_cases j; exacts [c35, c44])
    c9 c10 c14 c15 c16
    (Y := ![v56, v58]) (X := ![v57, v59]) (K := ![![v44, v53], ![v54, v55]])
    (fun j ↦ by fin_cases j; exacts [c48, c50])
    (fun j ↦ by fin_cases j; exacts [c49, c51])
    (fun j ↦ by fin_cases j; exacts [c36, c45])
    (fun j ↦ by fin_cases j; exacts [c46, c47])
    hswitch hμ hwalsh
  have hK : (fun k j ↦ ![![v6, v7], ![v8, v9]] k j + ![![v44, v53], ![v54, v55]] k j) =
      ![![matrixAdd v6 v44, matrixAdd v7 v53], ![matrixAdd v8 v54, matrixAdd v9 v55]] := by
    funext k j
    fin_cases k <;> fin_cases j <;> rfl
  have hR : (fun j ↦ ![v4, v5] j + (![v56, v58] j + ![v57, v59] j)) =
      ![matrixAdd v4 (matrixAdd v56 v57), matrixAdd v5 (matrixAdd v58 v59)] := by
    funext j
    fin_cases j <;> rfl
  rw [hK, hR] at hround
  exact hround

end Dio2Proof
