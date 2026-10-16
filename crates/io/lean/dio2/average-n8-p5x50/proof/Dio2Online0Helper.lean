import Dio2Lut
import Dio2Online0Refresh

/-!
# The Aut-helper targets of step 0

The refresh parts of step 0's `Aut(k)` helper pairs (`k = 3, 5, …, 15`; packed in the 36-family
of the stage output): each is a one-branch LUT (`W = 1`, its loop `sequential_generatedRoot_*`
runs zero times) of the σ sampler `w_56` at `σ_k`, re-targeted twice to the gadget, paired with
the one-public row `w_70` re-targeted to the gadget. Also the pair `(w_753, w_754)`: the σ
carrier `w_56_5` and `w_70`, both re-targeted to the gadget.
-/

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut Mxx.ExponentLut.Refresh RefreshTopProof RefreshTopProof.Gen RefreshTopProof.GenIface
  RefreshTopProof.GenFinal

/-- A loop run zero times ends where it starts. -/
theorem iterRuns_zero_eq {S : Type} {body : Nat → S → S → Prop} {c : Nat} {init final : S}
    (h : MxxIR.IterRuns body c init final) (hc : c = 0) : final = init := by
  subst hc
  cases h
  rfl

/-- A matrix re-encoded at its own modulus is itself. -/
theorem rebase_self {q r c : Nat} (hq : 1 < q) {G G' : ExactMatrix q 8 r c}
    (h : centeredRebaseRuns G G') : G' = G := by
  rw [centeredRebaseRuns_reduce h]
  exact reduceMatrix_centeredIntLift hq (by norm_num) G

open W_Stage_online_0 in
/-- The σ sampler's gadget `w_57` is the refresh gadget `w_229`. -/
theorem online0_gadget_eq {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs) :
    generatedRoot.w_57_0 ctx = generatedRoot.w_229_0 ctx :=
  gadgetMatrixRuns_unique F.c1.h57_0 F.c6.h229_0

set_option maxHeartbeats 32000000 in
/-- **An `Aut(k)` helper target, generically.** A one-branch LUT at `σ_k` of the input `(A, c)`
(an encoding of `μ (s_p, 1) G`), component `1` (`f₀, f₁` its switching ciphertexts), after a zero
loop (`Rl₁ = 0`), re-targeted to the gadget twice (`D₄`, `D₁₀`, digits of the gadget), paired with
the one-public row `c₁` re-targeted to the gadget: an encoding of `(Δ' ν, 1) G`. -/
theorem aut_target_gen {q : Nat} (hq1 : 1 < q) {backend : BackendContext} (layout : RegularLayout q)
    (hlayout : backend.regularLayout q 8 = some layout) (hexact : layout.droppedModuli = 0)
    (k : Int) (hodd : Odd k.toNat)
    {sm sp μ : ErrorPoly 8} (hsm : polyNorm sm ≤ 1)
    {G : ExactMatrix q 8 1 25} (hG : gadgetMatrixRuns backend 4096 25 G)
    {A : Fin 2 → Fin 2 → ExactMatrix q 8 1 25} {c : Fin 2 → ExactMatrix q 8 1 25} {B : Nat}
    (henc : Rel sm A (fun j ↦ reducePoly q 8 (((1 : Nat) : Int) * μ * unitPair sp j) • G) c B)
    {h3 : Fin 2 → ExactMatrix q 8 1 25} {Ad : Fin 2 → Fin 2 → ExactMatrix q 8 1 25} {Bh : Nat}
    (hhelp : Rel sm Ad (fun j ↦ reducePoly q 8 (((1 : Nat) : Int) *
      unitPair (substitution k.toNat hodd sm) j) • G) h3 Bh)
    {rc0 rA00 rA10 rc1 rA01 rA11 : ExactMatrix q 8 1 25}
    (hr0 : ringAutomorphismRuns k (c 0) rc0) (hr1 : ringAutomorphismRuns k (A 0 0) rA00)
    (hr2 : ringAutomorphismRuns k (A 1 0) rA10) (hr3 : ringAutomorphismRuns k (c 1) rc1)
    (hr4 : ringAutomorphismRuns k (A 0 1) rA01) (hr5 : ringAutomorphismRuns k (A 1 1) rA11)
    {l01 r01 : ExactMatrix q 8 (1 + 1) 25} {l r : ExactMatrix q 8 (1 + 1 + 1) 25}
    {both : ExactMatrix q 8 (1 + 1 + 1) (25 + 25)}
    (hl01 : concatRows rc0 rA00 l01) (hl : concatRows l01 rA10 l)
    (hr01 : concatRows rc1 rA01 r01) (hr : concatRows r01 rA11 r) (hboth : concatColumns l r both)
    {f0 f1 : ExactMatrix q 8 1 25} {Xb : ExactMatrix q 8 (1 + 1) 25} (hXb : concatRows f0 f1 Xb)
    {D' D : ExactMatrix q 8 (25 + 25) 25} (hD' : gadgetDecomposeRuns backend 4096 25 Xb D')
    (hD : compactCenteredRebaseRuns 2048 D' D)
    {M0 M1 : ExactMatrix q 8 1 25}
    (hM0 : sliceMatrix (matrixMul both D) 1 2 0 25 M0)
    (hM1 : sliceMatrix (matrixMul both D) 2 3 0 25 M1)
    {Z : ExactMatrix q 8 (1 + 1) 25} (hZ : concatRows M0 M1 Z)
    {E' E : ExactMatrix q 8 (25 + 25) 25} (hE' : gadgetDecomposeRuns backend 4096 25 Z E')
    (hE : compactCenteredRebaseRuns 2048 E' E)
    {hl01' hr01' : ExactMatrix q 8 (1 + 1) 25} {hl' hr' : ExactMatrix q 8 (1 + 1 + 1) 25}
    {H : ExactMatrix q 8 (1 + 1 + 1) (25 + 25)}
    (hhl01 : concatRows (h3 0) (Ad 0 0) hl01') (hhl : concatRows hl01' (Ad 1 0) hl')
    (hhr01 : concatRows (h3 1) (Ad 0 1) hr01') (hhr : concatRows hr01' (Ad 1 1) hr')
    (hH : concatColumns hl' hr' H)
    {Y X : ExactMatrix q 8 1 25}
    (hY : sliceMatrix (matrixMul H E) 0 1 0 25 Y)
    (hX : sliceMatrix (matrixMul both D) 0 1 0 25 X)
    {Rl1 : ExactMatrix q 8 1 25} (hRl1 : Rl1 = 0)
    {R0 : ExactMatrix q 8 1 25} (hR0 : centeredRebaseRuns G R0)
    {D4' D4 : ExactMatrix q 8 25 25}
    (hD4' : gadgetDecomposeRuns backend 4096 25 (matrixMulScalarRight R0 (matrixPolynomial [(1 : Int)])) D4')
    (hD4 : compactCenteredRebaseRuns 2048 D4' D4)
    {R1 : ExactMatrix q 8 1 25} (hR1 : centeredRebaseRuns G R1)
    {D10' D10 : ExactMatrix q 8 25 25}
    (hD10' : gadgetDecomposeRuns backend 4096 25 (matrixMulScalarRight R1 (matrixPolynomial [(1 : Int)])) D10')
    (hD10 : compactCenteredRebaseRuns 2048 D10' D10)
    {Δ' ν : ErrorPoly 8} {C : ExactPoly q 8} {BB : Nat}
    (hswitch : ∃ e : ErrorMatrix 8 1 25, CoeffBound e BB ∧
      reducePoly q 8 (substitution k.toNat hodd sp) • f0 + f1 =
        (C * reducePoly q 8 (Δ' * unitPair sp 1)) • G + reduceMatrix q 8 1 25 e)
    (hμ : polyNorm (substitution k.toNat hodd μ) ≤ 1)
    (hwalsh : reducePoly q 8 (substitution k.toNat hodd μ) * C = reducePoly q 8 ν)
    {c₁ : ExactMatrix q 8 1 25} {K₁ : Fin 2 → ExactMatrix q 8 1 25} {e₁ : ErrorMatrix 8 1 25}
    {Be1 : Nat} (he1 : CoeffBound e₁ Be1)
    (hc₁ : c₁ = maskPart (reducePoly q 8 sm) K₁ - G + reduceMatrix q 8 1 25 e₁) :
    Rel sm ![![rowOf (matrixMul H E) 1 * D4 * D10, K₁ 0 * D10],
        ![rowOf (matrixMul H E) 2 * D4 * D10, K₁ 1 * D10]]
      (fun j ↦ reducePoly q 8 (unitPair (Δ' * ν) j) • G)
      ![matrixMul (matrixMul (matrixAdd Rl1 (matrixAdd Y X)) D4) D10, matrixMul c₁ D10]
      (max (25 * 8 * (25 * 8 * ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048 + 8 * 1 * BB)
        * 2048) * 2048) (25 * 8 * Be1 * 2048)) := by
  -- The re-encoded gadget.
  have hg : centeredRebaseRuns G G := by
    have h := hR0
    rw [rebase_self hq1 hR0] at h
    exact h
  -- The LUT row.
  obtain ⟨eL, heL, hrow⟩ := lut_round_flat_at (n := 8) (by norm_num) 1 (log := 3) (by norm_num)
    (W := 1) (L := 0) rfl hq1 layout hlayout hexact (fun _ ↦ k) (fun _ ↦ hodd) hsm hG hg henc
    (fam := fun a b _ ↦ ![![f0, f0], ![f1, f1]] a b) (Kl := 0) (Rl := 0) (Bl := 0)
    (by rw [Finset.range_zero, Finset.sum_empty]; exact Rel.zero)
    hhelp hr0 hr1 hr2 hr3 hr4 hr5 hl01 hl hr01 hr hboth hXb hD' hD hM0 hM1 hZ hE' hE
    hhl01 hhl hhr01 hhr hH hY hX (C := fun _ ↦ C) (fun _ ↦ hswitch) (fun _ ↦ hμ)
    (by rw [Fin.sum_univ_one]; exact hwalsh)
  have hrow' : matrixAdd Rl1 (matrixAdd Y X) =
      maskPart (reducePoly q 8 sm) ![rowOf (matrixMul H E) 1, rowOf (matrixMul H E) 2] -
        reducePoly q 8 (Δ' * ν) • G + reduceMatrix q 8 1 25 eL := by
    rw [hRl1]
    have h0 : ((0 : Fin 2 → ExactMatrix q 8 1 25) 1) = 0 := rfl
    rw [h0] at hrow
    rw [hrow]
    simp [maskPart, unitPair]
  -- The digits of the gadget.
  have hX4 : matrixMulScalarRight R0 (matrixPolynomial [(1 : Int)]) = G := gadget_rebase_one hq1 hR0
  have hX10 : matrixMulScalarRight R1 (matrixPolynomial [(1 : Int)]) = G := gadget_rebase_one hq1 hR1
  rw [hX4] at hD4'
  rw [hX10] at hD10'
  have hGD4 := gadget_mul_decompose hq1 (by norm_num) layout hlayout hexact hG hD4'
  obtain ⟨d4, hd4, hdb4⟩ := gadgetDecomposeRuns_small hq1 (by norm_num) hD4'
  have hGD10 := gadget_mul_decompose hq1 (by norm_num) layout hlayout hexact hG hD10'
  obtain ⟨d10, hd10, hdb10⟩ := gadgetDecomposeRuns_small hq1 (by norm_num) hD10'
  have e4 : D4 = D4' := compactRebase_self_gen (by norm_num) hq1 hD4
  have e10 : D10 = D10' := compactRebase_self_gen (by norm_num) hq1 hD10
  rw [← e4] at hGD4 hd4
  rw [← e10] at hGD10 hd10
  refine ⟨![eL * d4 * d10, e₁ * d10], fun j ↦ ?_, fun j ↦ ?_⟩
  · fin_cases j
    · exact coeffBound_mono (coeffBound_mul (by norm_num) (coeffBound_mul (by norm_num) heL hdb4) hdb10)
        (le_trans (by simp) (le_max_left _ _))
    · exact coeffBound_mono (coeffBound_mul (by norm_num) he1 hdb10) (le_max_right _ _)
  · fin_cases j
    · show matrixAdd Rl1 (matrixAdd Y X) * D4 * D10 = _
      rw [hrow', Matrix.add_mul, Matrix.sub_mul, maskPart_mul, Matrix.smul_mul, hGD4, hd4,
        ← reduceMatrix_mul, Matrix.add_mul, Matrix.sub_mul, maskPart_mul, Matrix.smul_mul, hGD10, hd10,
        ← reduceMatrix_mul, ← hd4, ← hd10]
      simp [maskPart, unitPair]
    · show c₁ * D10 = _
      rw [hc₁, Matrix.add_mul, Matrix.sub_mul, maskPart_mul, hGD10, hd10, ← reduceMatrix_mul, ← hd10]
      simp [maskPart, unitPair]

open W_Stage_online_0 in
set_option maxHeartbeats 64000000 in
/-- **The `Aut(3)` helper target of step 0** (`(w_811, w_812)`): the one-branch LUT at
`σ_3` of the σ sampler (`henc`; helper `hhelp` on `w_764`/`w_765`, switching ciphertexts
`w_758`/`w_762` in `hswitch`) re-targeted to the gadget, with the one-public row `w_70`
(`hc₁`). -/
theorem online0_aut_3 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {sm sp μ : ErrorPoly 8} (hsm : polyNorm sm ≤ 1) {B : Nat}
    (henc : Rel sm ![![generatedRoot.w_56_0 ctx, generatedRoot.w_56_1 ctx], ![generatedRoot.w_56_2 ctx, generatedRoot.w_56_3 ctx]]
      (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * μ * unitPair sp j) • generatedRoot.w_57_0 ctx)
      ![generatedRoot.w_56_4 ctx, generatedRoot.w_56_5 ctx] B)
    {Bh : Nat}
    (hhelp : Rel sm (keyPairG hashModel (generatedRoot.w_7_0 ctx) modulus_0 25 helperPre [.integer 3])
      (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) *
        unitPair (substitution 3 (⟨1, rfl⟩ : Odd 3) sm) j) • generatedRoot.w_57_0 ctx)
      ![generatedRoot.w_764_0 ctx 0, generatedRoot.w_765_0 ctx 0] Bh)
    {Δ' ν : ErrorPoly 8} {C : ExactPoly modulus_0 8} {BB : Nat}
    (hswitch : ∃ e : ErrorMatrix 8 1 25, CoeffBound e BB ∧
      reducePoly modulus_0 8 (substitution 3 (⟨1, rfl⟩ : Odd 3) sp) • generatedRoot.w_758_0 ctx 0 +
        generatedRoot.w_762_0 ctx 0 = (C * reducePoly modulus_0 8 (Δ' * unitPair sp 1)) • generatedRoot.w_57_0 ctx +
        reduceMatrix modulus_0 8 1 25 e)
    (hμ : polyNorm (substitution 3 (⟨1, rfl⟩ : Odd 3) μ) ≤ 1)
    (hwalsh : reducePoly modulus_0 8 (substitution 3 (⟨1, rfl⟩ : Odd 3) μ) * C =
      reducePoly modulus_0 8 ν)
    {K₁ : Fin 2 → ExactMatrix modulus_0 8 1 25} {e₁ : ErrorMatrix 8 1 25} {Be1 : Nat}
    (he1 : CoeffBound e₁ Be1)
    (hc₁ : generatedRoot.w_70_0 ctx = maskPart (reducePoly modulus_0 8 sm) K₁ - generatedRoot.w_229_0 ctx +
      reduceMatrix modulus_0 8 1 25 e₁) :
    Rel sm ![![rowOf (matrixMul (generatedRoot.w_776_0 ctx) (generatedRoot.w_794_0 ctx)) 1 * generatedRoot.w_804_0 ctx * generatedRoot.w_810_0 ctx,
          K₁ 0 * generatedRoot.w_810_0 ctx],
        ![rowOf (matrixMul (generatedRoot.w_776_0 ctx) (generatedRoot.w_794_0 ctx)) 2 * generatedRoot.w_804_0 ctx * generatedRoot.w_810_0 ctx,
          K₁ 1 * generatedRoot.w_810_0 ctx]]
      (fun j ↦ reducePoly modulus_0 8 (unitPair (Δ' * ν) j) • generatedRoot.w_229_0 ctx)
      ![generatedRoot.w_811_0 ctx, generatedRoot.w_812_0 ctx]
      (max (25 * 8 * (25 * 8 * ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048 + 8 * 1 * BB)
        * 2048) * 2048) (25 * 8 * Be1 * 2048)) := by
  have hGG : generatedRoot.w_57_0 ctx = generatedRoot.w_229_0 ctx := gadgetMatrixRuns_unique F.c1.h57_0 F.c6.h229_0
  have h231 : generatedRoot.w_231_0 ctx = generatedRoot.w_229_0 ctx := rebase_left_one hq1 F.c6.h230_0
  have hget : ∀ {f : Fin 1 → ExactMatrix modulus_0 8 1 25} {x : ExactMatrix modulus_0 8 1 25},
      familyGetStatic f 0 x → x = f 0 := by
    intro f x hf
    obtain ⟨pos, hpos, rfl⟩ := hf
    congr 1
    exact Fin.ext (by omega)
  have hloop := iterRuns_zero_eq F.c27.h766_9 rfl
  have l1 : generatedRoot.w_766_1 ctx = 0 := congrArg (fun s ↦ s.2.1) hloop
  have l7 : generatedRoot.w_766_7 ctx = generatedRoot.w_758_0 ctx 0 :=
    (congrArg (fun s ↦ s.2.2.2.2.2.2.2.1) hloop).trans (hget F.c27.h759_0)
  have l9 : generatedRoot.w_766_9 ctx = generatedRoot.w_762_0 ctx 0 :=
    (congrArg (fun s ↦ s.2.2.2.2.2.2.2.2.2.1) hloop).trans (hget F.c27.h763_0)
  have hXb := F.c28.h786_0
  rw [l7, l9] at hXb
  have k00 := keyPairG_of_hash (pre := [.integer 3]) 0 0 (by norm_num) F.c27.h769_0
  have k10 := keyPairG_of_hash (pre := [.integer 3]) 1 0 (by norm_num) F.c27.h770_0
  have k01 := keyPairG_of_hash (pre := [.integer 3]) 0 1 (by norm_num) F.c27.h773_0
  have k11 := keyPairG_of_hash (pre := [.integer 3]) 1 1 (by norm_num) F.c27.h774_0
  have hkeys : keyPairG hashModel (generatedRoot.w_7_0 ctx) modulus_0 25 helperPre [.integer 3] =
      ![![generatedRoot.w_769_0 ctx, generatedRoot.w_773_0 ctx], ![generatedRoot.w_770_0 ctx, generatedRoot.w_774_0 ctx]] := by
    funext a j
    fin_cases a <;> fin_cases j
    exacts [k00.symm, k01.symm, k10.symm, k11.symm]
  have hrows : (![generatedRoot.w_764_0 ctx 0, generatedRoot.w_765_0 ctx 0] : Fin 2 → ExactMatrix modulus_0 8 1 25) =
      ![generatedRoot.w_767_0 ctx, generatedRoot.w_772_0 ctx] := by
    rw [← hget F.c27.h767_0, ← hget F.c27.h772_0]
  rw [hkeys, hrows, hGG] at hhelp
  rw [hGG] at henc hswitch
  have hR0 := F.c29.h800_0
  rw [hGG] at hR0
  have hR1 := F.c29.h806_0
  rw [h231] at hR1
  exact aut_target_gen hq1 layout hlayout hexact 3 (⟨1, rfl⟩ : Odd 3) hsm F.c6.h229_0 henc hhelp
    F.c27.h777_0 F.c27.h778_0 F.c27.h779_0 F.c28.h781_0 F.c28.h782_0 F.c28.h783_0
    F.c28.h778_0_1 F.c28.h780_0 F.c28.h782_0_1 F.c28.h784_0 F.c28.h785_0
    hXb F.c28.h787_0 F.c28.h788_0 F.c28.h790_0 F.c28.h791_0 F.c28.h792_0
    F.c29.h793_0 F.c29.h794_0
    F.c27.h769_0_1 F.c27.h771_0 F.c27.h773_0_1 F.c27.h775_0 F.c27.h776_0
    F.c29.h796_0 F.c29.h797_0 l1
    hR0 F.c29.h803_0 F.c29.h804_0 hR1 F.c29.h809_0 F.c29.h810_0
    hswitch hμ hwalsh he1 hc₁

open W_Stage_online_0 in
set_option maxHeartbeats 64000000 in
/-- **The `Aut(5)` helper target of step 0** (`(w_869, w_870)`): the one-branch LUT at
`σ_5` of the σ sampler (`henc`; helper `hhelp` on `w_822`/`w_823`, switching ciphertexts
`w_816`/`w_820` in `hswitch`) re-targeted to the gadget, with the one-public row `w_70`
(`hc₁`). -/
theorem online0_aut_5 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {sm sp μ : ErrorPoly 8} (hsm : polyNorm sm ≤ 1) {B : Nat}
    (henc : Rel sm ![![generatedRoot.w_56_0 ctx, generatedRoot.w_56_1 ctx], ![generatedRoot.w_56_2 ctx, generatedRoot.w_56_3 ctx]]
      (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * μ * unitPair sp j) • generatedRoot.w_57_0 ctx)
      ![generatedRoot.w_56_4 ctx, generatedRoot.w_56_5 ctx] B)
    {Bh : Nat}
    (hhelp : Rel sm (keyPairG hashModel (generatedRoot.w_7_0 ctx) modulus_0 25 helperPre [.integer 5])
      (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) *
        unitPair (substitution 5 (⟨2, rfl⟩ : Odd 5) sm) j) • generatedRoot.w_57_0 ctx)
      ![generatedRoot.w_822_0 ctx 0, generatedRoot.w_823_0 ctx 0] Bh)
    {Δ' ν : ErrorPoly 8} {C : ExactPoly modulus_0 8} {BB : Nat}
    (hswitch : ∃ e : ErrorMatrix 8 1 25, CoeffBound e BB ∧
      reducePoly modulus_0 8 (substitution 5 (⟨2, rfl⟩ : Odd 5) sp) • generatedRoot.w_816_0 ctx 0 +
        generatedRoot.w_820_0 ctx 0 = (C * reducePoly modulus_0 8 (Δ' * unitPair sp 1)) • generatedRoot.w_57_0 ctx +
        reduceMatrix modulus_0 8 1 25 e)
    (hμ : polyNorm (substitution 5 (⟨2, rfl⟩ : Odd 5) μ) ≤ 1)
    (hwalsh : reducePoly modulus_0 8 (substitution 5 (⟨2, rfl⟩ : Odd 5) μ) * C =
      reducePoly modulus_0 8 ν)
    {K₁ : Fin 2 → ExactMatrix modulus_0 8 1 25} {e₁ : ErrorMatrix 8 1 25} {Be1 : Nat}
    (he1 : CoeffBound e₁ Be1)
    (hc₁ : generatedRoot.w_70_0 ctx = maskPart (reducePoly modulus_0 8 sm) K₁ - generatedRoot.w_229_0 ctx +
      reduceMatrix modulus_0 8 1 25 e₁) :
    Rel sm ![![rowOf (matrixMul (generatedRoot.w_834_0 ctx) (generatedRoot.w_852_0 ctx)) 1 * generatedRoot.w_862_0 ctx * generatedRoot.w_868_0 ctx,
          K₁ 0 * generatedRoot.w_868_0 ctx],
        ![rowOf (matrixMul (generatedRoot.w_834_0 ctx) (generatedRoot.w_852_0 ctx)) 2 * generatedRoot.w_862_0 ctx * generatedRoot.w_868_0 ctx,
          K₁ 1 * generatedRoot.w_868_0 ctx]]
      (fun j ↦ reducePoly modulus_0 8 (unitPair (Δ' * ν) j) • generatedRoot.w_229_0 ctx)
      ![generatedRoot.w_869_0 ctx, generatedRoot.w_870_0 ctx]
      (max (25 * 8 * (25 * 8 * ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048 + 8 * 1 * BB)
        * 2048) * 2048) (25 * 8 * Be1 * 2048)) := by
  have hGG : generatedRoot.w_57_0 ctx = generatedRoot.w_229_0 ctx := gadgetMatrixRuns_unique F.c1.h57_0 F.c6.h229_0
  have h231 : generatedRoot.w_231_0 ctx = generatedRoot.w_229_0 ctx := rebase_left_one hq1 F.c6.h230_0
  have hget : ∀ {f : Fin 1 → ExactMatrix modulus_0 8 1 25} {x : ExactMatrix modulus_0 8 1 25},
      familyGetStatic f 0 x → x = f 0 := by
    intro f x hf
    obtain ⟨pos, hpos, rfl⟩ := hf
    congr 1
    exact Fin.ext (by omega)
  have hloop := iterRuns_zero_eq F.c29.h824_9 rfl
  have l1 : generatedRoot.w_824_1 ctx = 0 := congrArg (fun s ↦ s.2.1) hloop
  have l7 : generatedRoot.w_824_7 ctx = generatedRoot.w_816_0 ctx 0 :=
    (congrArg (fun s ↦ s.2.2.2.2.2.2.2.1) hloop).trans (hget F.c29.h817_0)
  have l9 : generatedRoot.w_824_9 ctx = generatedRoot.w_820_0 ctx 0 :=
    (congrArg (fun s ↦ s.2.2.2.2.2.2.2.2.2.1) hloop).trans (hget F.c29.h821_0)
  have hXb := F.c31.h844_0
  rw [l7, l9] at hXb
  have k00 := keyPairG_of_hash (pre := [.integer 5]) 0 0 (by norm_num) F.c29.h827_0
  have k10 := keyPairG_of_hash (pre := [.integer 5]) 1 0 (by norm_num) F.c30.h828_0
  have k01 := keyPairG_of_hash (pre := [.integer 5]) 0 1 (by norm_num) F.c30.h831_0
  have k11 := keyPairG_of_hash (pre := [.integer 5]) 1 1 (by norm_num) F.c30.h832_0
  have hkeys : keyPairG hashModel (generatedRoot.w_7_0 ctx) modulus_0 25 helperPre [.integer 5] =
      ![![generatedRoot.w_827_0 ctx, generatedRoot.w_831_0 ctx], ![generatedRoot.w_828_0 ctx, generatedRoot.w_832_0 ctx]] := by
    funext a j
    fin_cases a <;> fin_cases j
    exacts [k00.symm, k01.symm, k10.symm, k11.symm]
  have hrows : (![generatedRoot.w_822_0 ctx 0, generatedRoot.w_823_0 ctx 0] : Fin 2 → ExactMatrix modulus_0 8 1 25) =
      ![generatedRoot.w_825_0 ctx, generatedRoot.w_830_0 ctx] := by
    rw [← hget F.c29.h825_0, ← hget F.c30.h830_0]
  rw [hkeys, hrows, hGG] at hhelp
  rw [hGG] at henc hswitch
  have hR0 := F.c32.h858_0
  rw [hGG] at hR0
  have hR1 := F.c32.h864_0
  rw [h231] at hR1
  exact aut_target_gen hq1 layout hlayout hexact 5 (⟨2, rfl⟩ : Odd 5) hsm F.c6.h229_0 henc hhelp
    F.c30.h835_0 F.c30.h836_0 F.c30.h837_0 F.c30.h839_0 F.c30.h840_0 F.c30.h841_0
    F.c30.h836_0_1 F.c30.h838_0 F.c30.h840_0_1 F.c30.h842_0 F.c30.h843_0
    hXb F.c31.h845_0 F.c31.h846_0 F.c31.h848_0 F.c31.h849_0 F.c31.h850_0
    F.c31.h851_0 F.c31.h852_0
    F.c30.h827_0_1 F.c30.h829_0 F.c30.h831_0_1 F.c30.h833_0 F.c30.h834_0
    F.c31.h854_0 F.c32.h855_0 l1
    hR0 F.c32.h861_0 F.c32.h862_0 hR1 F.c32.h867_0 F.c32.h868_0
    hswitch hμ hwalsh he1 hc₁

open W_Stage_online_0 in
set_option maxHeartbeats 64000000 in
/-- **The `Aut(7)` helper target of step 0** (`(w_927, w_928)`): the one-branch LUT at
`σ_7` of the σ sampler (`henc`; helper `hhelp` on `w_880`/`w_881`, switching ciphertexts
`w_874`/`w_878` in `hswitch`) re-targeted to the gadget, with the one-public row `w_70`
(`hc₁`). -/
theorem online0_aut_7 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {sm sp μ : ErrorPoly 8} (hsm : polyNorm sm ≤ 1) {B : Nat}
    (henc : Rel sm ![![generatedRoot.w_56_0 ctx, generatedRoot.w_56_1 ctx], ![generatedRoot.w_56_2 ctx, generatedRoot.w_56_3 ctx]]
      (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * μ * unitPair sp j) • generatedRoot.w_57_0 ctx)
      ![generatedRoot.w_56_4 ctx, generatedRoot.w_56_5 ctx] B)
    {Bh : Nat}
    (hhelp : Rel sm (keyPairG hashModel (generatedRoot.w_7_0 ctx) modulus_0 25 helperPre [.integer 7])
      (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) *
        unitPair (substitution 7 (⟨3, rfl⟩ : Odd 7) sm) j) • generatedRoot.w_57_0 ctx)
      ![generatedRoot.w_880_0 ctx 0, generatedRoot.w_881_0 ctx 0] Bh)
    {Δ' ν : ErrorPoly 8} {C : ExactPoly modulus_0 8} {BB : Nat}
    (hswitch : ∃ e : ErrorMatrix 8 1 25, CoeffBound e BB ∧
      reducePoly modulus_0 8 (substitution 7 (⟨3, rfl⟩ : Odd 7) sp) • generatedRoot.w_874_0 ctx 0 +
        generatedRoot.w_878_0 ctx 0 = (C * reducePoly modulus_0 8 (Δ' * unitPair sp 1)) • generatedRoot.w_57_0 ctx +
        reduceMatrix modulus_0 8 1 25 e)
    (hμ : polyNorm (substitution 7 (⟨3, rfl⟩ : Odd 7) μ) ≤ 1)
    (hwalsh : reducePoly modulus_0 8 (substitution 7 (⟨3, rfl⟩ : Odd 7) μ) * C =
      reducePoly modulus_0 8 ν)
    {K₁ : Fin 2 → ExactMatrix modulus_0 8 1 25} {e₁ : ErrorMatrix 8 1 25} {Be1 : Nat}
    (he1 : CoeffBound e₁ Be1)
    (hc₁ : generatedRoot.w_70_0 ctx = maskPart (reducePoly modulus_0 8 sm) K₁ - generatedRoot.w_229_0 ctx +
      reduceMatrix modulus_0 8 1 25 e₁) :
    Rel sm ![![rowOf (matrixMul (generatedRoot.w_892_0 ctx) (generatedRoot.w_910_0 ctx)) 1 * generatedRoot.w_920_0 ctx * generatedRoot.w_926_0 ctx,
          K₁ 0 * generatedRoot.w_926_0 ctx],
        ![rowOf (matrixMul (generatedRoot.w_892_0 ctx) (generatedRoot.w_910_0 ctx)) 2 * generatedRoot.w_920_0 ctx * generatedRoot.w_926_0 ctx,
          K₁ 1 * generatedRoot.w_926_0 ctx]]
      (fun j ↦ reducePoly modulus_0 8 (unitPair (Δ' * ν) j) • generatedRoot.w_229_0 ctx)
      ![generatedRoot.w_927_0 ctx, generatedRoot.w_928_0 ctx]
      (max (25 * 8 * (25 * 8 * ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048 + 8 * 1 * BB)
        * 2048) * 2048) (25 * 8 * Be1 * 2048)) := by
  have hGG : generatedRoot.w_57_0 ctx = generatedRoot.w_229_0 ctx := gadgetMatrixRuns_unique F.c1.h57_0 F.c6.h229_0
  have h231 : generatedRoot.w_231_0 ctx = generatedRoot.w_229_0 ctx := rebase_left_one hq1 F.c6.h230_0
  have hget : ∀ {f : Fin 1 → ExactMatrix modulus_0 8 1 25} {x : ExactMatrix modulus_0 8 1 25},
      familyGetStatic f 0 x → x = f 0 := by
    intro f x hf
    obtain ⟨pos, hpos, rfl⟩ := hf
    congr 1
    exact Fin.ext (by omega)
  have hloop := iterRuns_zero_eq F.c32.h882_9 rfl
  have l1 : generatedRoot.w_882_1 ctx = 0 := congrArg (fun s ↦ s.2.1) hloop
  have l7 : generatedRoot.w_882_7 ctx = generatedRoot.w_874_0 ctx 0 :=
    (congrArg (fun s ↦ s.2.2.2.2.2.2.2.1) hloop).trans (hget F.c32.h875_0)
  have l9 : generatedRoot.w_882_9 ctx = generatedRoot.w_878_0 ctx 0 :=
    (congrArg (fun s ↦ s.2.2.2.2.2.2.2.2.2.1) hloop).trans (hget F.c32.h879_0)
  have hXb := F.c33.h902_0
  rw [l7, l9] at hXb
  have k00 := keyPairG_of_hash (pre := [.integer 7]) 0 0 (by norm_num) F.c32.h885_0
  have k10 := keyPairG_of_hash (pre := [.integer 7]) 1 0 (by norm_num) F.c32.h886_0
  have k01 := keyPairG_of_hash (pre := [.integer 7]) 0 1 (by norm_num) F.c32.h889_0
  have k11 := keyPairG_of_hash (pre := [.integer 7]) 1 1 (by norm_num) F.c32.h890_0
  have hkeys : keyPairG hashModel (generatedRoot.w_7_0 ctx) modulus_0 25 helperPre [.integer 7] =
      ![![generatedRoot.w_885_0 ctx, generatedRoot.w_889_0 ctx], ![generatedRoot.w_886_0 ctx, generatedRoot.w_890_0 ctx]] := by
    funext a j
    fin_cases a <;> fin_cases j
    exacts [k00.symm, k01.symm, k10.symm, k11.symm]
  have hrows : (![generatedRoot.w_880_0 ctx 0, generatedRoot.w_881_0 ctx 0] : Fin 2 → ExactMatrix modulus_0 8 1 25) =
      ![generatedRoot.w_883_0 ctx, generatedRoot.w_888_0 ctx] := by
    rw [← hget F.c32.h883_0, ← hget F.c32.h888_0]
  rw [hkeys, hrows, hGG] at hhelp
  rw [hGG] at henc hswitch
  have hR0 := F.c34.h916_0
  rw [hGG] at hR0
  have hR1 := F.c34.h922_0
  rw [h231] at hR1
  exact aut_target_gen hq1 layout hlayout hexact 7 (⟨3, rfl⟩ : Odd 7) hsm F.c6.h229_0 henc hhelp
    F.c32.h893_0 F.c33.h894_0 F.c33.h895_0 F.c33.h897_0 F.c33.h898_0 F.c33.h899_0
    F.c33.h894_0_1 F.c33.h896_0 F.c33.h898_0_1 F.c33.h900_0 F.c33.h901_0
    hXb F.c33.h903_0 F.c33.h904_0 F.c33.h906_0 F.c34.h907_0 F.c34.h908_0
    F.c34.h909_0 F.c34.h910_0
    F.c32.h885_0_1 F.c32.h887_0 F.c32.h889_0_1 F.c32.h891_0 F.c32.h892_0
    F.c34.h912_0 F.c34.h913_0 l1
    hR0 F.c34.h919_0 F.c34.h920_0 hR1 F.c34.h925_0 F.c34.h926_0
    hswitch hμ hwalsh he1 hc₁

open W_Stage_online_0 in
set_option maxHeartbeats 64000000 in
/-- **The `Aut(9)` helper target of step 0** (`(w_985, w_986)`): the one-branch LUT at
`σ_9` of the σ sampler (`henc`; helper `hhelp` on `w_938`/`w_939`, switching ciphertexts
`w_932`/`w_936` in `hswitch`) re-targeted to the gadget, with the one-public row `w_70`
(`hc₁`). -/
theorem online0_aut_9 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {sm sp μ : ErrorPoly 8} (hsm : polyNorm sm ≤ 1) {B : Nat}
    (henc : Rel sm ![![generatedRoot.w_56_0 ctx, generatedRoot.w_56_1 ctx], ![generatedRoot.w_56_2 ctx, generatedRoot.w_56_3 ctx]]
      (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * μ * unitPair sp j) • generatedRoot.w_57_0 ctx)
      ![generatedRoot.w_56_4 ctx, generatedRoot.w_56_5 ctx] B)
    {Bh : Nat}
    (hhelp : Rel sm (keyPairG hashModel (generatedRoot.w_7_0 ctx) modulus_0 25 helperPre [.integer 9])
      (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) *
        unitPair (substitution 9 (⟨4, rfl⟩ : Odd 9) sm) j) • generatedRoot.w_57_0 ctx)
      ![generatedRoot.w_938_0 ctx 0, generatedRoot.w_939_0 ctx 0] Bh)
    {Δ' ν : ErrorPoly 8} {C : ExactPoly modulus_0 8} {BB : Nat}
    (hswitch : ∃ e : ErrorMatrix 8 1 25, CoeffBound e BB ∧
      reducePoly modulus_0 8 (substitution 9 (⟨4, rfl⟩ : Odd 9) sp) • generatedRoot.w_932_0 ctx 0 +
        generatedRoot.w_936_0 ctx 0 = (C * reducePoly modulus_0 8 (Δ' * unitPair sp 1)) • generatedRoot.w_57_0 ctx +
        reduceMatrix modulus_0 8 1 25 e)
    (hμ : polyNorm (substitution 9 (⟨4, rfl⟩ : Odd 9) μ) ≤ 1)
    (hwalsh : reducePoly modulus_0 8 (substitution 9 (⟨4, rfl⟩ : Odd 9) μ) * C =
      reducePoly modulus_0 8 ν)
    {K₁ : Fin 2 → ExactMatrix modulus_0 8 1 25} {e₁ : ErrorMatrix 8 1 25} {Be1 : Nat}
    (he1 : CoeffBound e₁ Be1)
    (hc₁ : generatedRoot.w_70_0 ctx = maskPart (reducePoly modulus_0 8 sm) K₁ - generatedRoot.w_229_0 ctx +
      reduceMatrix modulus_0 8 1 25 e₁) :
    Rel sm ![![rowOf (matrixMul (generatedRoot.w_950_0 ctx) (generatedRoot.w_968_0 ctx)) 1 * generatedRoot.w_978_0 ctx * generatedRoot.w_984_0 ctx,
          K₁ 0 * generatedRoot.w_984_0 ctx],
        ![rowOf (matrixMul (generatedRoot.w_950_0 ctx) (generatedRoot.w_968_0 ctx)) 2 * generatedRoot.w_978_0 ctx * generatedRoot.w_984_0 ctx,
          K₁ 1 * generatedRoot.w_984_0 ctx]]
      (fun j ↦ reducePoly modulus_0 8 (unitPair (Δ' * ν) j) • generatedRoot.w_229_0 ctx)
      ![generatedRoot.w_985_0 ctx, generatedRoot.w_986_0 ctx]
      (max (25 * 8 * (25 * 8 * ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048 + 8 * 1 * BB)
        * 2048) * 2048) (25 * 8 * Be1 * 2048)) := by
  have hGG : generatedRoot.w_57_0 ctx = generatedRoot.w_229_0 ctx := gadgetMatrixRuns_unique F.c1.h57_0 F.c6.h229_0
  have h231 : generatedRoot.w_231_0 ctx = generatedRoot.w_229_0 ctx := rebase_left_one hq1 F.c6.h230_0
  have hget : ∀ {f : Fin 1 → ExactMatrix modulus_0 8 1 25} {x : ExactMatrix modulus_0 8 1 25},
      familyGetStatic f 0 x → x = f 0 := by
    intro f x hf
    obtain ⟨pos, hpos, rfl⟩ := hf
    congr 1
    exact Fin.ext (by omega)
  have hloop := iterRuns_zero_eq F.c34.h940_9 rfl
  have l1 : generatedRoot.w_940_1 ctx = 0 := congrArg (fun s ↦ s.2.1) hloop
  have l7 : generatedRoot.w_940_7 ctx = generatedRoot.w_932_0 ctx 0 :=
    (congrArg (fun s ↦ s.2.2.2.2.2.2.2.1) hloop).trans (hget F.c34.h933_0)
  have l9 : generatedRoot.w_940_9 ctx = generatedRoot.w_936_0 ctx 0 :=
    (congrArg (fun s ↦ s.2.2.2.2.2.2.2.2.2.1) hloop).trans (hget F.c34.h937_0)
  have hXb := F.c36.h960_0
  rw [l7, l9] at hXb
  have k00 := keyPairG_of_hash (pre := [.integer 9]) 0 0 (by norm_num) F.c35.h943_0
  have k10 := keyPairG_of_hash (pre := [.integer 9]) 1 0 (by norm_num) F.c35.h944_0
  have k01 := keyPairG_of_hash (pre := [.integer 9]) 0 1 (by norm_num) F.c35.h947_0
  have k11 := keyPairG_of_hash (pre := [.integer 9]) 1 1 (by norm_num) F.c35.h948_0
  have hkeys : keyPairG hashModel (generatedRoot.w_7_0 ctx) modulus_0 25 helperPre [.integer 9] =
      ![![generatedRoot.w_943_0 ctx, generatedRoot.w_947_0 ctx], ![generatedRoot.w_944_0 ctx, generatedRoot.w_948_0 ctx]] := by
    funext a j
    fin_cases a <;> fin_cases j
    exacts [k00.symm, k01.symm, k10.symm, k11.symm]
  have hrows : (![generatedRoot.w_938_0 ctx 0, generatedRoot.w_939_0 ctx 0] : Fin 2 → ExactMatrix modulus_0 8 1 25) =
      ![generatedRoot.w_941_0 ctx, generatedRoot.w_946_0 ctx] := by
    rw [← hget F.c35.h941_0, ← hget F.c35.h946_0]
  rw [hkeys, hrows, hGG] at hhelp
  rw [hGG] at henc hswitch
  have hR0 := F.c37.h974_0
  rw [hGG] at hR0
  have hR1 := F.c37.h980_0
  rw [h231] at hR1
  exact aut_target_gen hq1 layout hlayout hexact 9 (⟨4, rfl⟩ : Odd 9) hsm F.c6.h229_0 henc hhelp
    F.c35.h951_0 F.c35.h952_0 F.c35.h953_0 F.c35.h955_0 F.c35.h956_0 F.c36.h957_0
    F.c35.h952_0_1 F.c35.h954_0 F.c36.h956_0_1 F.c36.h958_0 F.c36.h959_0
    hXb F.c36.h961_0 F.c36.h962_0 F.c36.h964_0 F.c36.h965_0 F.c36.h966_0
    F.c36.h967_0 F.c36.h968_0
    F.c35.h943_0_1 F.c35.h945_0 F.c35.h947_0_1 F.c35.h949_0 F.c35.h950_0
    F.c36.h970_0 F.c37.h971_0 l1
    hR0 F.c37.h977_0 F.c37.h978_0 hR1 F.c37.h983_0 F.c37.h984_0
    hswitch hμ hwalsh he1 hc₁

open W_Stage_online_0 in
set_option maxHeartbeats 64000000 in
/-- **The `Aut(11)` helper target of step 0** (`(w_1043, w_1044)`): the one-branch LUT at
`σ_11` of the σ sampler (`henc`; helper `hhelp` on `w_996`/`w_997`, switching ciphertexts
`w_990`/`w_994` in `hswitch`) re-targeted to the gadget, with the one-public row `w_70`
(`hc₁`). -/
theorem online0_aut_11 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {sm sp μ : ErrorPoly 8} (hsm : polyNorm sm ≤ 1) {B : Nat}
    (henc : Rel sm ![![generatedRoot.w_56_0 ctx, generatedRoot.w_56_1 ctx], ![generatedRoot.w_56_2 ctx, generatedRoot.w_56_3 ctx]]
      (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * μ * unitPair sp j) • generatedRoot.w_57_0 ctx)
      ![generatedRoot.w_56_4 ctx, generatedRoot.w_56_5 ctx] B)
    {Bh : Nat}
    (hhelp : Rel sm (keyPairG hashModel (generatedRoot.w_7_0 ctx) modulus_0 25 helperPre [.integer 11])
      (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) *
        unitPair (substitution 11 (⟨5, rfl⟩ : Odd 11) sm) j) • generatedRoot.w_57_0 ctx)
      ![generatedRoot.w_996_0 ctx 0, generatedRoot.w_997_0 ctx 0] Bh)
    {Δ' ν : ErrorPoly 8} {C : ExactPoly modulus_0 8} {BB : Nat}
    (hswitch : ∃ e : ErrorMatrix 8 1 25, CoeffBound e BB ∧
      reducePoly modulus_0 8 (substitution 11 (⟨5, rfl⟩ : Odd 11) sp) • generatedRoot.w_990_0 ctx 0 +
        generatedRoot.w_994_0 ctx 0 = (C * reducePoly modulus_0 8 (Δ' * unitPair sp 1)) • generatedRoot.w_57_0 ctx +
        reduceMatrix modulus_0 8 1 25 e)
    (hμ : polyNorm (substitution 11 (⟨5, rfl⟩ : Odd 11) μ) ≤ 1)
    (hwalsh : reducePoly modulus_0 8 (substitution 11 (⟨5, rfl⟩ : Odd 11) μ) * C =
      reducePoly modulus_0 8 ν)
    {K₁ : Fin 2 → ExactMatrix modulus_0 8 1 25} {e₁ : ErrorMatrix 8 1 25} {Be1 : Nat}
    (he1 : CoeffBound e₁ Be1)
    (hc₁ : generatedRoot.w_70_0 ctx = maskPart (reducePoly modulus_0 8 sm) K₁ - generatedRoot.w_229_0 ctx +
      reduceMatrix modulus_0 8 1 25 e₁) :
    Rel sm ![![rowOf (matrixMul (generatedRoot.w_1008_0 ctx) (generatedRoot.w_1026_0 ctx)) 1 * generatedRoot.w_1036_0 ctx * generatedRoot.w_1042_0 ctx,
          K₁ 0 * generatedRoot.w_1042_0 ctx],
        ![rowOf (matrixMul (generatedRoot.w_1008_0 ctx) (generatedRoot.w_1026_0 ctx)) 2 * generatedRoot.w_1036_0 ctx * generatedRoot.w_1042_0 ctx,
          K₁ 1 * generatedRoot.w_1042_0 ctx]]
      (fun j ↦ reducePoly modulus_0 8 (unitPair (Δ' * ν) j) • generatedRoot.w_229_0 ctx)
      ![generatedRoot.w_1043_0 ctx, generatedRoot.w_1044_0 ctx]
      (max (25 * 8 * (25 * 8 * ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048 + 8 * 1 * BB)
        * 2048) * 2048) (25 * 8 * Be1 * 2048)) := by
  have hGG : generatedRoot.w_57_0 ctx = generatedRoot.w_229_0 ctx := gadgetMatrixRuns_unique F.c1.h57_0 F.c6.h229_0
  have h231 : generatedRoot.w_231_0 ctx = generatedRoot.w_229_0 ctx := rebase_left_one hq1 F.c6.h230_0
  have hget : ∀ {f : Fin 1 → ExactMatrix modulus_0 8 1 25} {x : ExactMatrix modulus_0 8 1 25},
      familyGetStatic f 0 x → x = f 0 := by
    intro f x hf
    obtain ⟨pos, hpos, rfl⟩ := hf
    congr 1
    exact Fin.ext (by omega)
  have hloop := iterRuns_zero_eq F.c37.h998_9 rfl
  have l1 : generatedRoot.w_998_1 ctx = 0 := congrArg (fun s ↦ s.2.1) hloop
  have l7 : generatedRoot.w_998_7 ctx = generatedRoot.w_990_0 ctx 0 :=
    (congrArg (fun s ↦ s.2.2.2.2.2.2.2.1) hloop).trans (hget F.c37.h991_0)
  have l9 : generatedRoot.w_998_9 ctx = generatedRoot.w_994_0 ctx 0 :=
    (congrArg (fun s ↦ s.2.2.2.2.2.2.2.2.2.1) hloop).trans (hget F.c37.h995_0)
  have hXb := F.c38.h1018_0
  rw [l7, l9] at hXb
  have k00 := keyPairG_of_hash (pre := [.integer 11]) 0 0 (by norm_num) F.c37.h1001_0
  have k10 := keyPairG_of_hash (pre := [.integer 11]) 1 0 (by norm_num) F.c37.h1002_0
  have k01 := keyPairG_of_hash (pre := [.integer 11]) 0 1 (by norm_num) F.c37.h1005_0
  have k11 := keyPairG_of_hash (pre := [.integer 11]) 1 1 (by norm_num) F.c37.h1006_0
  have hkeys : keyPairG hashModel (generatedRoot.w_7_0 ctx) modulus_0 25 helperPre [.integer 11] =
      ![![generatedRoot.w_1001_0 ctx, generatedRoot.w_1005_0 ctx], ![generatedRoot.w_1002_0 ctx, generatedRoot.w_1006_0 ctx]] := by
    funext a j
    fin_cases a <;> fin_cases j
    exacts [k00.symm, k01.symm, k10.symm, k11.symm]
  have hrows : (![generatedRoot.w_996_0 ctx 0, generatedRoot.w_997_0 ctx 0] : Fin 2 → ExactMatrix modulus_0 8 1 25) =
      ![generatedRoot.w_999_0 ctx, generatedRoot.w_1004_0 ctx] := by
    rw [← hget F.c37.h999_0, ← hget F.c37.h1004_0]
  rw [hkeys, hrows, hGG] at hhelp
  rw [hGG] at henc hswitch
  have hR0 := F.c39.h1032_0
  rw [hGG] at hR0
  have hR1 := F.c39.h1038_0
  rw [h231] at hR1
  exact aut_target_gen hq1 layout hlayout hexact 11 (⟨5, rfl⟩ : Odd 11) hsm F.c6.h229_0 henc hhelp
    F.c38.h1009_0 F.c38.h1010_0 F.c38.h1011_0 F.c38.h1013_0 F.c38.h1014_0 F.c38.h1015_0
    F.c38.h1010_0_1 F.c38.h1012_0 F.c38.h1014_0_1 F.c38.h1016_0 F.c38.h1017_0
    hXb F.c38.h1019_0 F.c38.h1020_0 F.c38.h1022_0 F.c39.h1023_0 F.c39.h1024_0
    F.c39.h1025_0 F.c39.h1026_0
    F.c37.h1001_0_1 F.c37.h1003_0 F.c37.h1005_0_1 F.c37.h1007_0 F.c37.h1008_0
    F.c39.h1028_0 F.c39.h1029_0 l1
    hR0 F.c39.h1035_0 F.c39.h1036_0 hR1 F.c39.h1041_0 F.c39.h1042_0
    hswitch hμ hwalsh he1 hc₁

open W_Stage_online_0 in
set_option maxHeartbeats 64000000 in
/-- **The `Aut(13)` helper target of step 0** (`(w_1101, w_1102)`): the one-branch LUT at
`σ_13` of the σ sampler (`henc`; helper `hhelp` on `w_1054`/`w_1055`, switching ciphertexts
`w_1048`/`w_1052` in `hswitch`) re-targeted to the gadget, with the one-public row `w_70`
(`hc₁`). -/
theorem online0_aut_13 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {sm sp μ : ErrorPoly 8} (hsm : polyNorm sm ≤ 1) {B : Nat}
    (henc : Rel sm ![![generatedRoot.w_56_0 ctx, generatedRoot.w_56_1 ctx], ![generatedRoot.w_56_2 ctx, generatedRoot.w_56_3 ctx]]
      (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * μ * unitPair sp j) • generatedRoot.w_57_0 ctx)
      ![generatedRoot.w_56_4 ctx, generatedRoot.w_56_5 ctx] B)
    {Bh : Nat}
    (hhelp : Rel sm (keyPairG hashModel (generatedRoot.w_7_0 ctx) modulus_0 25 helperPre [.integer 13])
      (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) *
        unitPair (substitution 13 (⟨6, rfl⟩ : Odd 13) sm) j) • generatedRoot.w_57_0 ctx)
      ![generatedRoot.w_1054_0 ctx 0, generatedRoot.w_1055_0 ctx 0] Bh)
    {Δ' ν : ErrorPoly 8} {C : ExactPoly modulus_0 8} {BB : Nat}
    (hswitch : ∃ e : ErrorMatrix 8 1 25, CoeffBound e BB ∧
      reducePoly modulus_0 8 (substitution 13 (⟨6, rfl⟩ : Odd 13) sp) • generatedRoot.w_1048_0 ctx 0 +
        generatedRoot.w_1052_0 ctx 0 = (C * reducePoly modulus_0 8 (Δ' * unitPair sp 1)) • generatedRoot.w_57_0 ctx +
        reduceMatrix modulus_0 8 1 25 e)
    (hμ : polyNorm (substitution 13 (⟨6, rfl⟩ : Odd 13) μ) ≤ 1)
    (hwalsh : reducePoly modulus_0 8 (substitution 13 (⟨6, rfl⟩ : Odd 13) μ) * C =
      reducePoly modulus_0 8 ν)
    {K₁ : Fin 2 → ExactMatrix modulus_0 8 1 25} {e₁ : ErrorMatrix 8 1 25} {Be1 : Nat}
    (he1 : CoeffBound e₁ Be1)
    (hc₁ : generatedRoot.w_70_0 ctx = maskPart (reducePoly modulus_0 8 sm) K₁ - generatedRoot.w_229_0 ctx +
      reduceMatrix modulus_0 8 1 25 e₁) :
    Rel sm ![![rowOf (matrixMul (generatedRoot.w_1066_0 ctx) (generatedRoot.w_1084_0 ctx)) 1 * generatedRoot.w_1094_0 ctx * generatedRoot.w_1100_0 ctx,
          K₁ 0 * generatedRoot.w_1100_0 ctx],
        ![rowOf (matrixMul (generatedRoot.w_1066_0 ctx) (generatedRoot.w_1084_0 ctx)) 2 * generatedRoot.w_1094_0 ctx * generatedRoot.w_1100_0 ctx,
          K₁ 1 * generatedRoot.w_1100_0 ctx]]
      (fun j ↦ reducePoly modulus_0 8 (unitPair (Δ' * ν) j) • generatedRoot.w_229_0 ctx)
      ![generatedRoot.w_1101_0 ctx, generatedRoot.w_1102_0 ctx]
      (max (25 * 8 * (25 * 8 * ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048 + 8 * 1 * BB)
        * 2048) * 2048) (25 * 8 * Be1 * 2048)) := by
  have hGG : generatedRoot.w_57_0 ctx = generatedRoot.w_229_0 ctx := gadgetMatrixRuns_unique F.c1.h57_0 F.c6.h229_0
  have h231 : generatedRoot.w_231_0 ctx = generatedRoot.w_229_0 ctx := rebase_left_one hq1 F.c6.h230_0
  have hget : ∀ {f : Fin 1 → ExactMatrix modulus_0 8 1 25} {x : ExactMatrix modulus_0 8 1 25},
      familyGetStatic f 0 x → x = f 0 := by
    intro f x hf
    obtain ⟨pos, hpos, rfl⟩ := hf
    congr 1
    exact Fin.ext (by omega)
  have hloop := iterRuns_zero_eq F.c40.h1056_9 rfl
  have l1 : generatedRoot.w_1056_1 ctx = 0 := congrArg (fun s ↦ s.2.1) hloop
  have l7 : generatedRoot.w_1056_7 ctx = generatedRoot.w_1048_0 ctx 0 :=
    (congrArg (fun s ↦ s.2.2.2.2.2.2.2.1) hloop).trans (hget F.c39.h1049_0)
  have l9 : generatedRoot.w_1056_9 ctx = generatedRoot.w_1052_0 ctx 0 :=
    (congrArg (fun s ↦ s.2.2.2.2.2.2.2.2.2.1) hloop).trans (hget F.c40.h1053_0)
  have hXb := F.c41.h1076_0
  rw [l7, l9] at hXb
  have k00 := keyPairG_of_hash (pre := [.integer 13]) 0 0 (by norm_num) F.c40.h1059_0
  have k10 := keyPairG_of_hash (pre := [.integer 13]) 1 0 (by norm_num) F.c40.h1060_0
  have k01 := keyPairG_of_hash (pre := [.integer 13]) 0 1 (by norm_num) F.c40.h1063_0
  have k11 := keyPairG_of_hash (pre := [.integer 13]) 1 1 (by norm_num) F.c40.h1064_0
  have hkeys : keyPairG hashModel (generatedRoot.w_7_0 ctx) modulus_0 25 helperPre [.integer 13] =
      ![![generatedRoot.w_1059_0 ctx, generatedRoot.w_1063_0 ctx], ![generatedRoot.w_1060_0 ctx, generatedRoot.w_1064_0 ctx]] := by
    funext a j
    fin_cases a <;> fin_cases j
    exacts [k00.symm, k01.symm, k10.symm, k11.symm]
  have hrows : (![generatedRoot.w_1054_0 ctx 0, generatedRoot.w_1055_0 ctx 0] : Fin 2 → ExactMatrix modulus_0 8 1 25) =
      ![generatedRoot.w_1057_0 ctx, generatedRoot.w_1062_0 ctx] := by
    rw [← hget F.c40.h1057_0, ← hget F.c40.h1062_0]
  rw [hkeys, hrows, hGG] at hhelp
  rw [hGG] at henc hswitch
  have hR0 := F.c42.h1090_0
  rw [hGG] at hR0
  have hR1 := F.c42.h1096_0
  rw [h231] at hR1
  exact aut_target_gen hq1 layout hlayout hexact 13 (⟨6, rfl⟩ : Odd 13) hsm F.c6.h229_0 henc hhelp
    F.c40.h1067_0 F.c40.h1068_0 F.c40.h1069_0 F.c40.h1071_0 F.c41.h1072_0 F.c41.h1073_0
    F.c40.h1068_0_1 F.c40.h1070_0 F.c41.h1072_0_1 F.c41.h1074_0 F.c41.h1075_0
    hXb F.c41.h1077_0 F.c41.h1078_0 F.c41.h1080_0 F.c41.h1081_0 F.c41.h1082_0
    F.c41.h1083_0 F.c41.h1084_0
    F.c40.h1059_0_1 F.c40.h1061_0 F.c40.h1063_0_1 F.c40.h1065_0 F.c40.h1066_0
    F.c42.h1086_0 F.c42.h1087_0 l1
    hR0 F.c42.h1093_0 F.c42.h1094_0 hR1 F.c42.h1099_0 F.c42.h1100_0
    hswitch hμ hwalsh he1 hc₁

open W_Stage_online_0 in
set_option maxHeartbeats 64000000 in
/-- **The `Aut(15)` helper target of step 0** (`(w_1159, w_1160)`): the one-branch LUT at
`σ_15` of the σ sampler (`henc`; helper `hhelp` on `w_1112`/`w_1113`, switching ciphertexts
`w_1106`/`w_1110` in `hswitch`) re-targeted to the gadget, with the one-public row `w_70`
(`hc₁`). -/
theorem online0_aut_15 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {sm sp μ : ErrorPoly 8} (hsm : polyNorm sm ≤ 1) {B : Nat}
    (henc : Rel sm ![![generatedRoot.w_56_0 ctx, generatedRoot.w_56_1 ctx], ![generatedRoot.w_56_2 ctx, generatedRoot.w_56_3 ctx]]
      (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * μ * unitPair sp j) • generatedRoot.w_57_0 ctx)
      ![generatedRoot.w_56_4 ctx, generatedRoot.w_56_5 ctx] B)
    {Bh : Nat}
    (hhelp : Rel sm (keyPairG hashModel (generatedRoot.w_7_0 ctx) modulus_0 25 helperPre [.integer 15])
      (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) *
        unitPair (substitution 15 (⟨7, rfl⟩ : Odd 15) sm) j) • generatedRoot.w_57_0 ctx)
      ![generatedRoot.w_1112_0 ctx 0, generatedRoot.w_1113_0 ctx 0] Bh)
    {Δ' ν : ErrorPoly 8} {C : ExactPoly modulus_0 8} {BB : Nat}
    (hswitch : ∃ e : ErrorMatrix 8 1 25, CoeffBound e BB ∧
      reducePoly modulus_0 8 (substitution 15 (⟨7, rfl⟩ : Odd 15) sp) • generatedRoot.w_1106_0 ctx 0 +
        generatedRoot.w_1110_0 ctx 0 = (C * reducePoly modulus_0 8 (Δ' * unitPair sp 1)) • generatedRoot.w_57_0 ctx +
        reduceMatrix modulus_0 8 1 25 e)
    (hμ : polyNorm (substitution 15 (⟨7, rfl⟩ : Odd 15) μ) ≤ 1)
    (hwalsh : reducePoly modulus_0 8 (substitution 15 (⟨7, rfl⟩ : Odd 15) μ) * C =
      reducePoly modulus_0 8 ν)
    {K₁ : Fin 2 → ExactMatrix modulus_0 8 1 25} {e₁ : ErrorMatrix 8 1 25} {Be1 : Nat}
    (he1 : CoeffBound e₁ Be1)
    (hc₁ : generatedRoot.w_70_0 ctx = maskPart (reducePoly modulus_0 8 sm) K₁ - generatedRoot.w_229_0 ctx +
      reduceMatrix modulus_0 8 1 25 e₁) :
    Rel sm ![![rowOf (matrixMul (generatedRoot.w_1124_0 ctx) (generatedRoot.w_1142_0 ctx)) 1 * generatedRoot.w_1152_0 ctx * generatedRoot.w_1158_0 ctx,
          K₁ 0 * generatedRoot.w_1158_0 ctx],
        ![rowOf (matrixMul (generatedRoot.w_1124_0 ctx) (generatedRoot.w_1142_0 ctx)) 2 * generatedRoot.w_1152_0 ctx * generatedRoot.w_1158_0 ctx,
          K₁ 1 * generatedRoot.w_1158_0 ctx]]
      (fun j ↦ reducePoly modulus_0 8 (unitPair (Δ' * ν) j) • generatedRoot.w_229_0 ctx)
      ![generatedRoot.w_1159_0 ctx, generatedRoot.w_1160_0 ctx]
      (max (25 * 8 * (25 * 8 * ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048 + 8 * 1 * BB)
        * 2048) * 2048) (25 * 8 * Be1 * 2048)) := by
  have hGG : generatedRoot.w_57_0 ctx = generatedRoot.w_229_0 ctx := gadgetMatrixRuns_unique F.c1.h57_0 F.c6.h229_0
  have h231 : generatedRoot.w_231_0 ctx = generatedRoot.w_229_0 ctx := rebase_left_one hq1 F.c6.h230_0
  have hget : ∀ {f : Fin 1 → ExactMatrix modulus_0 8 1 25} {x : ExactMatrix modulus_0 8 1 25},
      familyGetStatic f 0 x → x = f 0 := by
    intro f x hf
    obtain ⟨pos, hpos, rfl⟩ := hf
    congr 1
    exact Fin.ext (by omega)
  have hloop := iterRuns_zero_eq F.c42.h1114_9 rfl
  have l1 : generatedRoot.w_1114_1 ctx = 0 := congrArg (fun s ↦ s.2.1) hloop
  have l7 : generatedRoot.w_1114_7 ctx = generatedRoot.w_1106_0 ctx 0 :=
    (congrArg (fun s ↦ s.2.2.2.2.2.2.2.1) hloop).trans (hget F.c42.h1107_0)
  have l9 : generatedRoot.w_1114_9 ctx = generatedRoot.w_1110_0 ctx 0 :=
    (congrArg (fun s ↦ s.2.2.2.2.2.2.2.2.2.1) hloop).trans (hget F.c42.h1111_0)
  have hXb := F.c43.h1134_0
  rw [l7, l9] at hXb
  have k00 := keyPairG_of_hash (pre := [.integer 15]) 0 0 (by norm_num) F.c42.h1117_0
  have k10 := keyPairG_of_hash (pre := [.integer 15]) 1 0 (by norm_num) F.c42.h1118_0
  have k01 := keyPairG_of_hash (pre := [.integer 15]) 0 1 (by norm_num) F.c42.h1121_0
  have k11 := keyPairG_of_hash (pre := [.integer 15]) 1 1 (by norm_num) F.c42.h1122_0
  have hkeys : keyPairG hashModel (generatedRoot.w_7_0 ctx) modulus_0 25 helperPre [.integer 15] =
      ![![generatedRoot.w_1117_0 ctx, generatedRoot.w_1121_0 ctx], ![generatedRoot.w_1118_0 ctx, generatedRoot.w_1122_0 ctx]] := by
    funext a j
    fin_cases a <;> fin_cases j
    exacts [k00.symm, k01.symm, k10.symm, k11.symm]
  have hrows : (![generatedRoot.w_1112_0 ctx 0, generatedRoot.w_1113_0 ctx 0] : Fin 2 → ExactMatrix modulus_0 8 1 25) =
      ![generatedRoot.w_1115_0 ctx, generatedRoot.w_1120_0 ctx] := by
    rw [← hget F.c42.h1115_0, ← hget F.c42.h1120_0]
  rw [hkeys, hrows, hGG] at hhelp
  rw [hGG] at henc hswitch
  have hR0 := F.c44.h1148_0
  rw [hGG] at hR0
  have hR1 := F.c44.h1154_0
  rw [h231] at hR1
  exact aut_target_gen hq1 layout hlayout hexact 15 (⟨7, rfl⟩ : Odd 15) hsm F.c6.h229_0 henc hhelp
    F.c43.h1125_0 F.c43.h1126_0 F.c43.h1127_0 F.c43.h1129_0 F.c43.h1130_0 F.c43.h1131_0
    F.c43.h1126_0_1 F.c43.h1128_0 F.c43.h1130_0_1 F.c43.h1132_0 F.c43.h1133_0
    hXb F.c43.h1135_0 F.c43.h1136_0 F.c44.h1138_0 F.c44.h1139_0 F.c44.h1140_0
    F.c44.h1141_0 F.c44.h1142_0
    F.c42.h1117_0_1 F.c42.h1119_0 F.c43.h1121_0_1 F.c43.h1123_0 F.c43.h1124_0
    F.c44.h1144_0 F.c44.h1145_0 l1
    hR0 F.c44.h1151_0 F.c44.h1152_0 hR1 F.c45.h1157_0 F.c45.h1158_0
    hswitch hμ hwalsh he1 hc₁

open W_Stage_online_0 in
set_option maxHeartbeats 32000000 in
/-- **The σ-carrier target pair of step 0** (`(w_753, w_754)`): component `1` of the σ sampler
total (`hσ`, as in `online0_g`) acted on by the digits `w_746` of the gadget, and the one-public
row `w_70` (`hc₁`), both re-targeted to the gadget by `w_752`: an encoding of
`(Δ' Σ_o X^o ν_o, 1) G`. -/
theorem online0_target_sigma {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {sm sp Δ' : ErrorPoly 8} {ν : Fin 8 → ErrorPoly 8} {B : Nat}
    (hσ : EncRel sm (generatedRoot.w_56_0 ctx, (generatedRoot.w_56_1 ctx, (generatedRoot.w_56_2 ctx,
        (generatedRoot.w_56_3 ctx, (generatedRoot.w_56_4 ctx, (generatedRoot.w_56_5 ctx, ()))))))
      (∑ o : Fin 8, fun j ↦ reducePoly modulus_0 8 (Xe (n := 8) (o : Nat)) •
        (reducePoly modulus_0 8 (Δ' * ν o * unitPair sp j) • generatedRoot.w_57_0 ctx)) B)
    {K₁ : Fin 2 → ExactMatrix modulus_0 8 1 25} {e₁ : ErrorMatrix 8 1 25} {Be1 : Nat}
    (he1 : CoeffBound e₁ Be1)
    (hc₁ : generatedRoot.w_70_0 ctx = maskPart (reducePoly modulus_0 8 sm) K₁ - generatedRoot.w_229_0 ctx +
      reduceMatrix modulus_0 8 1 25 e₁) :
    Rel sm ![![generatedRoot.w_56_1 ctx * generatedRoot.w_746_0 ctx * generatedRoot.w_752_0 ctx,
          K₁ 0 * generatedRoot.w_752_0 ctx],
        ![generatedRoot.w_56_3 ctx * generatedRoot.w_746_0 ctx * generatedRoot.w_752_0 ctx,
          K₁ 1 * generatedRoot.w_752_0 ctx]]
      (fun j ↦ reducePoly modulus_0 8 (unitPair (Δ' * ∑ o : Fin 8, Xe (n := 8) (o : Nat) * ν o) j) •
        generatedRoot.w_229_0 ctx)
      ![generatedRoot.w_753_0 ctx, generatedRoot.w_754_0 ctx]
      (max (25 * 8 * (25 * 8 * B * 2048) * 2048) (25 * 8 * Be1 * 2048)) := by
  have hGG : generatedRoot.w_57_0 ctx = generatedRoot.w_229_0 ctx :=
    gadgetMatrixRuns_unique F.c1.h57_0 F.c6.h229_0
  have h231 : generatedRoot.w_231_0 ctx = generatedRoot.w_229_0 ctx := rebase_left_one hq1 F.c6.h230_0
  -- The σ carrier at the gadget's digits.
  have hX : generatedRoot.w_744_0 ctx = generatedRoot.w_57_0 ctx := gadget_rebase_one hq1 F.c26.h742_0
  have hD : generatedRoot.w_746_0 ctx = generatedRoot.w_745_0 ctx :=
    compactRebase_self_gen (by norm_num) hq1 F.c27.h746_0
  have h745 := F.c26.h745_0
  rw [hX] at h745
  obtain ⟨e, he, hc⟩ := sampler_gamma_one (by norm_num) hq1 layout hlayout hexact F.c1.h57_0 h745 hσ
  rw [← hD, hGG] at hc
  have hA : (fun k ↦ Enc.A (generatedRoot.w_56_0 ctx, (generatedRoot.w_56_1 ctx,
      (generatedRoot.w_56_2 ctx, (generatedRoot.w_56_3 ctx, (generatedRoot.w_56_4 ctx,
        (generatedRoot.w_56_5 ctx, ())))))) k 1 * generatedRoot.w_746_0 ctx) =
      ![generatedRoot.w_56_1 ctx * generatedRoot.w_746_0 ctx,
        generatedRoot.w_56_3 ctx * generatedRoot.w_746_0 ctx] := by
    funext k
    fin_cases k <;> rfl
  rw [hA] at hc
  -- The digits of the gadget.
  have h748 := F.c27.h748_0
  rw [h231] at h748
  have hX2 : generatedRoot.w_750_0 ctx = generatedRoot.w_229_0 ctx := gadget_rebase_one hq1 h748
  have hD2 : generatedRoot.w_752_0 ctx = generatedRoot.w_751_0 ctx :=
    compactRebase_self_gen (by norm_num) hq1 F.c27.h752_0
  have h751 := F.c27.h751_0
  rw [hX2] at h751
  have hGD := gadget_mul_decompose hq1 (by norm_num) layout hlayout hexact F.c6.h229_0 h751
  obtain ⟨d, hd, hdb⟩ := gadgetDecomposeRuns_small hq1 (by norm_num) h751
  rw [← hD2] at hGD hd
  refine ⟨![e * d, e₁ * d], fun j ↦ ?_, fun j ↦ ?_⟩
  · fin_cases j
    · exact coeffBound_mono (coeffBound_mul (by norm_num) he hdb) (le_max_left _ _)
    · exact coeffBound_mono (coeffBound_mul (by norm_num) he1 hdb) (le_max_right _ _)
  · fin_cases j
    · show generatedRoot.w_56_5 ctx * generatedRoot.w_746_0 ctx * generatedRoot.w_752_0 ctx = _
      rw [show generatedRoot.w_56_5 ctx * generatedRoot.w_746_0 ctx = _ from hc, Matrix.add_mul, Matrix.sub_mul, maskPart_mul, Matrix.smul_mul, hGD, hd,
        ← reduceMatrix_mul, ← hd]
      simp [maskPart, unitPair]
    · show generatedRoot.w_70_0 ctx * generatedRoot.w_752_0 ctx = _
      rw [hc₁, Matrix.add_mul, Matrix.sub_mul, maskPart_mul, hGD, hd, ← reduceMatrix_mul, ← hd]
      simp [maskPart, unitPair]

end Dio2Proof
