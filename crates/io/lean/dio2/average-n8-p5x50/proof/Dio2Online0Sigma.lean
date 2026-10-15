import Dio2Online1Prf
import Gen_W_Stage_online_0_Scopes
import W_Stage_online_0_Facts

/-!
# The σ sampler of step 0's online stage

The same chain as step 1's smudge PRF (`Dio2Online1Prf`): the first bucket (static cells
`parallel_generatedRoot_29`, row sums `parallel_generatedRoot_30`), the second bucket (dynamic
cells `parallel_generatedRoot_33`, row sums `parallel_generatedRoot_34`), then per member the
readout LUT (`parallel_generatedRoot_35`), the sampling LUT (`parallel_generatedRoot_36`, whose
input is the readout's output), the placement at `X^o` (`parallel_generatedRoot_37`, role
`sampler.monomial`: a plain `multiplyMonomial` by `o`) and the member total
(`sequential_generatedRoot_56`).
-/

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof.Gen RefreshTopProof.GenIface
  RefreshTopProof.GenInst W_Stage_online_0

/-- The tag prefix of the step-0 σ sampler's first-bucket selector tops
(`rns-bgg/rhs/RhsId(3)/0/1`). -/
abbrev rhsPre0 : Blob :=
  [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 51, 41, 47,
    48, 47, 49]

/-- The tag prefix of the step-0 σ sampler's second-bucket selector tops
(`rns-bgg/rhs/RhsId(4)/1/2`). -/
abbrev rhsPre0b : Blob :=
  [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 52, 41, 47,
    49, 47, 50]

/-- The parameter and Program identities in the step-0 selector tags. -/
abbrev prfA0 : Int := 29270270146341990167195247313302370525046749248022274291861926128744555509284
abbrev prfB0 : Int := 60863387405588526063623438538454545359885565288667817947515354790518700822262

set_option maxHeartbeats 16000000 in
/-- **The first bucket of the step-0 σ sampler.** Cell `i` multiplies label member `t_i` (the
gather table `w_18`) by the static state and by its selector right-hand side; member `o` sums its
two cells. -/
theorem online0_bucket0 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {sm sp μ : ErrorPoly 8} {α : Fin 2 → ErrorPoly 8} (hα : ∀ k, polyNorm (α k) ≤ 1)
    (hαμ : ∀ k, polyNorm (α k * μ) ≤ 1)
    {G : ExactMatrix modulus_0 8 1 25} (hG : gadgetMatrixRuns backend 4096 25 G) {Ba : Nat}
    (ha : ∀ k : Fin 2, Rel sm (labelKeysG hashModel (generatedRoot.w_7_0 ctx) modulus_0 25 labelPre k)
      (fun j ↦ reducePoly modulus_0 8 (α k * unitPair sm j) • G)
      ![generatedRoot.w_16_0 ctx k, generatedRoot.w_17_0 ctx k] Ba)
    (hpay : ∀ k, generatedRoot.w_22_0 ctx k = fun _ _ ↦ reducePoly modulus_0 8 (α k))
    {g' g : ExactMatrix modulus_0 8 1 25} (hg' : gadgetMatrixRuns backend 4096 25 g')
    (hg : centeredRebaseRuns g' g) {Bs : Nat}
    (hs : Rel sm ![![generatedRoot.w_19_0 ctx, generatedRoot.w_23_0 ctx],
        ![generatedRoot.w_20_0 ctx, generatedRoot.w_24_0 ctx]]
      (fun j ↦ reducePoly modulus_0 8 (μ * unitPair sp j) • g)
      ![generatedRoot.w_21_0 ctx, generatedRoot.w_25_0 ctx] Bs)
    {ν : Fin 16 → ErrorPoly 8} {hT : ExactMatrix modulus_0 8 1 25}
    {eB : Fin 16 → Fin 2 → ErrorMatrix 8 1 25} {BB : Nat}
    (hrhs : ∀ (i : Fin 16) j, reducePoly modulus_0 8 sp •
        topPairG hashModel (generatedRoot.w_26_0 ctx) modulus_0 25 rhsPre0 (prfTopTags prfA0 prfB0 i) j +
        ![generatedRoot.w_27_0 ctx i, generatedRoot.w_28_0 ctx i] j =
      reducePoly modulus_0 8 (((1 : Nat) : Int) * ν i * unitPair sp j) • hT +
        reduceMatrix modulus_0 8 1 25 (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB)
    (sel : Fin 8 → Fin 2) (hsel : ∀ o j, ν (cellAt o j) = if j = sel o then (1 : ErrorPoly 8) else 0)
    (o : Fin 8) :
    ∃ key : Fin 2, ((key : Nat) : Int) = generatedRoot.w_18_0 ctx (cellAt o (sel o)) ∧
      EncRel sm (generatedRoot.w_30_0 ctx o, (generatedRoot.w_30_1 ctx o,
        (generatedRoot.w_30_2 ctx o, (generatedRoot.w_30_3 ctx o, (generatedRoot.w_30_4 ctx o,
          (generatedRoot.w_30_5 ctx o, ()))))))
        (fun i ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * (α key * μ) * unitPair sp i) • hT)
        (2 * ((25 + 25) * 8 * ((25 + 25) * 8 * Ba * 2048 + 8 * 1 * Bs) * 2048 + 8 * 1 * BB)) := by
  have hcell : ∀ (o : Fin 8) (j : Fin 2), ∃ k : Fin 2,
      ((k : Nat) : Int) = generatedRoot.w_18_0 ctx (cellAt o j) ∧
      EncRel sm (Tup.ev (k := 6) (generatedRoot.w_29_0 ctx, (generatedRoot.w_29_1 ctx,
          (generatedRoot.w_29_2 ctx, (generatedRoot.w_29_3 ctx, (generatedRoot.w_29_4 ctx,
            (generatedRoot.w_29_5 ctx, ())))))) (cellAt o j))
        (fun i ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * (α k * μ * ν (cellAt o j)) *
          unitPair sp i) • hT)
        ((25 + 25) * 8 * ((25 + 25) * 8 * Ba * 2048 + 8 * 1 * Bs) * 2048 + 8 * 1 * BB) := by
    intro o j
    obtain ⟨x0, x1, xp, -, -, g0, -, -, g1, -, -, g2, hc⟩ := F.c0.h29_5 (cellAt o j)
    obtain ⟨k, hk, rfl⟩ := g0
    obtain ⟨k1, hk1, rfl⟩ := g1
    obtain ⟨k2, hk2, rfl⟩ := g2
    have e1 : k = k1 := Fin.ext (by omega)
    have e2 : k = k2 := Fin.ext (by omega)
    subst e1 e2
    refine ⟨k, hk, ?_⟩
    have hak := ha k
    rw [hk] at hak
    have hrel := PrfCellFlat.rel (by norm_num) hq1 layout hlayout hexact
      (parallel_generatedRoot_29.sem hc) hG (hα k) (hαμ k) hak (hpay k) hg' hg hs
      (hrhs (cellAt o j)) (heB _)
    refine encRel_of_rel (hrel.target_eq ?_) (fun a j ↦ by fin_cases a <;> fin_cases j <;> rfl)
      (fun j ↦ by fin_cases j <;> rfl)
    funext i
    rw [smul_smul, ← map_mul]
    congr 2
    ring
  obtain ⟨key, hkey, -⟩ := hcell o (sel o)
  exact ⟨key, hkey, bucket_of_cells (M := 8) (W := 2) (K := 2) (p := 1) α (fun _ ↦ μ)
    (generatedRoot.w_18_0 ctx) ν hcell o (parallel_generatedRoot_30.sem (F.c0.h30_5 o))
    (sel o) (hsel o) key hkey⟩

set_option maxHeartbeats 16000000 in
/-- **The second bucket of the step-0 σ sampler.** Cell `i` multiplies label member `t'_i` (the
gather table `w_15`) by the state of member `⌊i / 2⌋` (the first bucket's output, `hstate`) and by
its selector right-hand side; member `o` encodes `α_key μ_o`. -/
theorem online0_bucket1 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {sm sp : ErrorPoly 8} {μ : Fin 8 → ErrorPoly 8} {α : Fin 2 → ErrorPoly 8}
    (hα : ∀ k, polyNorm (α k) ≤ 1) (hαμ : ∀ k o, polyNorm (α k * μ o) ≤ 1)
    {G : ExactMatrix modulus_0 8 1 25} (hG : gadgetMatrixRuns backend 4096 25 G) {Ba : Nat}
    (ha : ∀ k : Fin 2, Rel sm (labelKeysG hashModel (generatedRoot.w_7_0 ctx) modulus_0 25 labelPre k)
      (fun j ↦ reducePoly modulus_0 8 (α k * unitPair sm j) • G)
      ![generatedRoot.w_16_0 ctx k, generatedRoot.w_17_0 ctx k] Ba)
    (hpay : ∀ k, generatedRoot.w_22_0 ctx k = fun _ _ ↦ reducePoly modulus_0 8 (α k))
    {g' g : ExactMatrix modulus_0 8 1 25} (hg' : gadgetMatrixRuns backend 4096 25 g')
    (hg : centeredRebaseRuns g' g) {Bs : Nat}
    (hstate : ∀ o : Fin 8, EncRel sm (generatedRoot.w_30_0 ctx o, (generatedRoot.w_30_1 ctx o,
        (generatedRoot.w_30_2 ctx o, (generatedRoot.w_30_3 ctx o, (generatedRoot.w_30_4 ctx o,
          (generatedRoot.w_30_5 ctx o, ()))))))
      (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * μ o * unitPair sp j) • g) Bs)
    {ν : Fin 16 → ErrorPoly 8} {hT : ExactMatrix modulus_0 8 1 25}
    {eB : Fin 16 → Fin 2 → ErrorMatrix 8 1 25} {BB : Nat}
    (hrhs : ∀ (i : Fin 16) j, reducePoly modulus_0 8 sp •
        topPairG hashModel (generatedRoot.w_26_0 ctx) modulus_0 25 rhsPre0b (prfTopTags prfA0 prfB0 i) j +
        ![generatedRoot.w_31_0 ctx i, generatedRoot.w_32_0 ctx i] j =
      reducePoly modulus_0 8 (((1 : Nat) : Int) * ν i * unitPair sp j) • hT +
        reduceMatrix modulus_0 8 1 25 (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB)
    (sel : Fin 8 → Fin 2) (hsel : ∀ o j, ν (cellAt o j) = if j = sel o then (1 : ErrorPoly 8) else 0)
    (o : Fin 8) :
    ∃ key : Fin 2, ((key : Nat) : Int) = generatedRoot.w_15_0 ctx (cellAt o (sel o)) ∧
      EncRel sm (generatedRoot.w_34_0 ctx o, (generatedRoot.w_34_1 ctx o,
        (generatedRoot.w_34_2 ctx o, (generatedRoot.w_34_3 ctx o, (generatedRoot.w_34_4 ctx o,
          (generatedRoot.w_34_5 ctx o, ()))))))
        (fun i ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * (α key * μ o) * unitPair sp i) • hT)
        (2 * ((25 + 25) * 8 * ((25 + 25) * 8 * Ba * 2048 + 8 * 1 * Bs) * 2048 + 8 * 1 * BB)) := by
  have hcell : ∀ (o : Fin 8) (j : Fin 2), ∃ k : Fin 2,
      ((k : Nat) : Int) = generatedRoot.w_15_0 ctx (cellAt o j) ∧
      EncRel sm (Tup.ev (k := 6) (generatedRoot.w_33_0 ctx, (generatedRoot.w_33_1 ctx,
          (generatedRoot.w_33_2 ctx, (generatedRoot.w_33_3 ctx, (generatedRoot.w_33_4 ctx,
            (generatedRoot.w_33_5 ctx, ())))))) (cellAt o j))
        (fun i ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * (α k * μ o * ν (cellAt o j)) *
          unitPair sp i) • hT)
        ((25 + 25) * 8 * ((25 + 25) * 8 * Ba * 2048 + 8 * 1 * Bs) * 2048 + 8 * 1 * BB) := by
    intro o j
    obtain ⟨x0, x1, xp, -, -, g0, -, -, g1, -, -, g2, hc⟩ := F.c0.h33_5 (cellAt o j)
    obtain ⟨k, hk, rfl⟩ := g0
    obtain ⟨k1, hk1, rfl⟩ := g1
    obtain ⟨k2, hk2, rfl⟩ := g2
    have e1 : k = k1 := Fin.ext (by omega)
    have e2 : k = k2 := Fin.ext (by omega)
    subst e1 e2
    refine ⟨k, hk, ?_⟩
    have hak := ha k
    rw [hk] at hak
    have hio : (cellAt o j : Nat) / 2 < 8 := by
      have := o.isLt; have := j.isLt; simp only [cellAt]; omega
    have hoo : (⟨(cellAt o j : Nat) / 2, hio⟩ : Fin 8) = o := by
      apply Fin.ext; have := j.isLt; simp only [cellAt]; omega
    have hs := hstate o
    rw [← hoo] at hs
    unfold EncRel at hs
    have hrel := PrfCellFlat.rel_1 (by norm_num) hq1 layout hlayout hexact
      (parallel_generatedRoot_33.sem hc) hio hG (hα k) (by rw [hoo]; exact hαμ k o) hak (hpay k)
      hg' hg (μ := μ ⟨(cellAt o j : Nat) / 2, hio⟩) (hs.target_eq (by funext j; simp))
      (hrhs (cellAt o j)) (heB _)
    rw [hoo] at hrel
    refine encRel_of_rel (hrel.target_eq ?_) (fun a j ↦ by fin_cases a <;> fin_cases j <;> rfl)
      (fun j ↦ by fin_cases j <;> rfl)
    funext i
    rw [smul_smul, ← map_mul]
    congr 2
    ring
  obtain ⟨key, hkey, -⟩ := hcell o (sel o)
  exact ⟨key, hkey, bucket_of_cells (M := 8) (W := 2) (K := 2) (p := 1) α μ
    (generatedRoot.w_15_0 ctx) ν hcell o (parallel_generatedRoot_34.sem (F.c0.h34_5 o))
    (sel o) (hsel o) key hkey⟩

/-- **Placement by a plain monomial** (role `sampler.monomial`): every part of the encoding is
multiplied by `X^o` (no `fmod`), so it encodes `X^o • t`. -/
theorem monomial_place_rel {n : Nat} (hn : 1 < n) {q m : Nat} {sm : ErrorPoly n} {o : Nat}
    (ho : o < 2 * n) {x0 x1 x2 x3 x4 x5 : ExactMatrix q n 1 m}
    {out : Tup (ExactMatrix q n 1 m) 6}
    (h : out = (multiplyMonomial x0 (Int.ofNat o), (multiplyMonomial x1 (Int.ofNat o),
      (multiplyMonomial x2 (Int.ofNat o), (multiplyMonomial x3 (Int.ofNat o),
        (multiplyMonomial x4 (Int.ofNat o), (multiplyMonomial x5 (Int.ofNat o), ())))))))
    {t : Fin 2 → ExactMatrix q n 1 m} {B : Nat}
    (hin : EncRel sm (x0, (x1, (x2, (x3, (x4, (x5, ())))))) t B) :
    EncRel sm out (fun j ↦ reducePoly q n (Xe (n := n) o) • t j) B := by
  subst h
  have hr := rel_monomial_gen hn hin ho
  have hA : Enc.A (multiplyMonomial x0 (Int.ofNat o), (multiplyMonomial x1 (Int.ofNat o),
      (multiplyMonomial x2 (Int.ofNat o), (multiplyMonomial x3 (Int.ofNat o),
        (multiplyMonomial x4 (Int.ofNat o), (multiplyMonomial x5 (Int.ofNat o), ())))))) =
      fun a j ↦ multiplyMonomial (Enc.A (x0, (x1, (x2, (x3, (x4, (x5, ())))))) a j) (o : Int) := by
    funext a j; fin_cases a <;> fin_cases j <;> rfl
  have hc : Enc.c (multiplyMonomial x0 (Int.ofNat o), (multiplyMonomial x1 (Int.ofNat o),
      (multiplyMonomial x2 (Int.ofNat o), (multiplyMonomial x3 (Int.ofNat o),
        (multiplyMonomial x4 (Int.ofNat o), (multiplyMonomial x5 (Int.ofNat o), ())))))) =
      fun j ↦ multiplyMonomial (Enc.c (x0, (x1, (x2, (x3, (x4, (x5, ())))))) j) (o : Int) := by
    funext j; fin_cases j <;> rfl
  unfold EncRel
  rw [hA, hc]
  exact hr

set_option maxHeartbeats 64000000 in
/-- **The step-0 σ sampler's LUTs, placed and summed.** Member `o` of the second bucket encodes
`μ_o` (`hin`); its readout LUT (`parallel_generatedRoot_35`) encodes `Δr ν_o` against `g`
(`LutMemberFlat.rel`, LUT identity `hwalsh o`); the sampling LUT (`parallel_generatedRoot_36`)
reads it and encodes `Δ' νs_o` against `gq` (LUT identity `hwalshS o`); the result is placed at
`X^o` (`parallel_generatedRoot_37`) and summed over the eight members
(`sequential_generatedRoot_56`). -/
theorem online0_sigma_total {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {sm sp : ErrorPoly 8} (hsm : polyNorm sm ≤ 1)
    {g' g : ExactMatrix modulus_0 8 1 25} (hg' : gadgetMatrixRuns backend 4096 25 g')
    (hg : centeredRebaseRuns g' g) {μ : Fin 8 → ErrorPoly 8} {B : Nat}
    (hin : ∀ o : Fin 8, EncRel sm (generatedRoot.w_34_0 ctx o, (generatedRoot.w_34_1 ctx o,
        (generatedRoot.w_34_2 ctx o, (generatedRoot.w_34_3 ctx o, (generatedRoot.w_34_4 ctx o,
          (generatedRoot.w_34_5 ctx o, ()))))))
      (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * μ o * unitPair sp j) • g) B)
    -- The readout LUT: helpers `w_13`/`w_14`, switching ciphertexts `w_9 … w_12`.
    {Bh : Nat}
    (hhelp : ∀ i : Fin 8, Rel sm (keyPairG hashModel (generatedRoot.w_7_0 ctx) modulus_0 25 helperPre
        [.integer (sigmaOf sigmas8 i)])
      (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) *
        unitPair (substitution (sigmaOf sigmas8 i).toNat (sigmas8_odd i) sm) j) • g)
      ![generatedRoot.w_13_0 ctx i, generatedRoot.w_14_0 ctx i] Bh)
    {Δr : ErrorPoly 8} {ν : Fin 8 → ErrorPoly 8}
    {C : Fin 8 → ExactPoly modulus_0 8} {BB : Nat}
    (hswitch : ∀ i : Fin 8, ∃ e : Fin 2 → ErrorMatrix 8 1 25, (∀ j, CoeffBound (e j) BB) ∧ ∀ j,
      reducePoly modulus_0 8 (substitution (sigmaOf sigmas8 i).toNat (sigmas8_odd i) sp) •
          ![![generatedRoot.w_9_0 ctx, generatedRoot.w_10_0 ctx],
            ![generatedRoot.w_11_0 ctx, generatedRoot.w_12_0 ctx]] 0 j i +
        ![![generatedRoot.w_9_0 ctx, generatedRoot.w_10_0 ctx],
          ![generatedRoot.w_11_0 ctx, generatedRoot.w_12_0 ctx]] 1 j i =
      (C i * reducePoly modulus_0 8 (Δr * unitPair sp j)) • g + reduceMatrix modulus_0 8 1 25 (e j))
    (hμ : ∀ o i, polyNorm (substitution (sigmaOf sigmas8 i).toNat (sigmas8_odd i) (μ o)) ≤ 1)
    (hwalsh : ∀ o, ∑ i, reducePoly modulus_0 8 (substitution (sigmaOf sigmas8 i).toNat
      (sigmas8_odd i) (μ o)) * C i = reducePoly modulus_0 8 (ν o))
    -- The sampling LUT: helpers `w_6`/`w_8`, switching ciphertexts `w_2 … w_5`.
    {Bh' : Nat}
    (hhelpS : ∀ i : Fin 8, Rel sm (keyPairG hashModel (generatedRoot.w_7_0 ctx) modulus_0 25 helperPre
        [.integer (sigmaOf sigmas8 i)])
      (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) *
        unitPair (substitution (sigmaOf sigmas8 i).toNat (sigmas8_odd i) sm) j) • g)
      ![generatedRoot.w_6_0 ctx i, generatedRoot.w_8_0 ctx i] Bh')
    {Δ' : ErrorPoly 8} {νs : Fin 8 → ErrorPoly 8} {gq : ExactMatrix modulus_0 8 1 25}
    {CS : Fin 8 → ExactPoly modulus_0 8} {BB' : Nat}
    (hswitchS : ∀ i : Fin 8, ∃ e : Fin 2 → ErrorMatrix 8 1 25, (∀ j, CoeffBound (e j) BB') ∧ ∀ j,
      reducePoly modulus_0 8 (substitution (sigmaOf sigmas8 i).toNat (sigmas8_odd i) sp) •
          ![![generatedRoot.w_2_0 ctx, generatedRoot.w_3_0 ctx],
            ![generatedRoot.w_4_0 ctx, generatedRoot.w_5_0 ctx]] 0 j i +
        ![![generatedRoot.w_2_0 ctx, generatedRoot.w_3_0 ctx],
          ![generatedRoot.w_4_0 ctx, generatedRoot.w_5_0 ctx]] 1 j i =
      (CS i * reducePoly modulus_0 8 (Δ' * unitPair sp j)) • gq +
        reduceMatrix modulus_0 8 1 25 (e j))
    (hμS : ∀ o i, polyNorm (substitution (sigmaOf sigmas8 i).toNat (sigmas8_odd i)
      (Δr * ν o)) ≤ 1)
    (hwalshS : ∀ o, ∑ i, reducePoly modulus_0 8 (substitution (sigmaOf sigmas8 i).toNat
      (sigmas8_odd i) (Δr * ν o)) * CS i = reducePoly modulus_0 8 (νs o)) :
    EncRel sm (generatedRoot.w_56_0 ctx, (generatedRoot.w_56_1 ctx, (generatedRoot.w_56_2 ctx,
        (generatedRoot.w_56_3 ctx, (generatedRoot.w_56_4 ctx, (generatedRoot.w_56_5 ctx, ()))))))
      (∑ o : Fin 8, fun j ↦ reducePoly modulus_0 8 (Xe (n := 8) o) •
        (reducePoly modulus_0 8 (Δ' * νs o * unitPair sp j) • gq))
      (8 * (7 * ((25 + 25) * 8 * Bh' * 2048 + (25 + 25) * 8 *
          (7 * ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048) +
            ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048) + 8 * (8 * 1 * BB)) * 2048) +
        ((25 + 25) * 8 * Bh' * 2048 + (25 + 25) * 8 *
          (7 * ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048) +
            ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048) + 8 * (8 * 1 * BB)) * 2048) +
        8 * (8 * 1 * BB'))) := by
  -- Every member's readout LUT.
  have hmem : ∀ o : Fin 8, EncRel sm (generatedRoot.w_35_0 ctx o, (generatedRoot.w_35_1 ctx o,
      (generatedRoot.w_35_2 ctx o, (generatedRoot.w_35_3 ctx o, (generatedRoot.w_35_4 ctx o,
        (generatedRoot.w_35_5 ctx o, ()))))))
      (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * (Δr * ν o) * unitPair sp j) • g)
      (7 * ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048) +
        ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048) + 8 * (8 * 1 * BB)) := by
    intro o
    have he := hin o
    unfold EncRel at he
    have hr := LutMemberFlat.rel (by norm_num) (log := 3) (by norm_num) (L := 7) (by norm_num) hq1
      layout hlayout hexact (fun i a b hb ↦ sequential_parallel_generatedRoot_35_18.sem hb)
      (parallel_generatedRoot_35.sem (F.c0.h35_5 o)) sigmas8_odd hsm hg' hg he
      hhelp hswitch (hμ o) (hwalsh o)
    refine encRel_of_rel (hr.target_eq ?_) (fun a j ↦ by fin_cases a <;> fin_cases j <;> rfl)
      (fun j ↦ by fin_cases j <;> rfl)
    funext j
    congr 2
    push_cast
    ring
  -- Every member's sampling LUT, reading the readout.
  have hsamp : ∀ o : Fin 8, EncRel sm (generatedRoot.w_36_0 ctx o, (generatedRoot.w_36_1 ctx o,
      (generatedRoot.w_36_2 ctx o, (generatedRoot.w_36_3 ctx o, (generatedRoot.w_36_4 ctx o,
        (generatedRoot.w_36_5 ctx o, ()))))))
      (fun j ↦ reducePoly modulus_0 8 (Δ' * νs o * unitPair sp j) • gq)
      (7 * ((25 + 25) * 8 * Bh' * 2048 + (25 + 25) * 8 *
          (7 * ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048) +
            ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048) + 8 * (8 * 1 * BB)) * 2048) +
        ((25 + 25) * 8 * Bh' * 2048 + (25 + 25) * 8 *
          (7 * ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048) +
            ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048) + 8 * (8 * 1 * BB)) * 2048) +
        8 * (8 * 1 * BB')) := by
    intro o
    have he := hmem o
    unfold EncRel at he
    have hr := LutMemberFlat.rel (by norm_num) (log := 3) (by norm_num) (L := 7) (by norm_num) hq1
      layout hlayout hexact (fun i a b hb ↦ sequential_parallel_generatedRoot_36_18.sem hb)
      (parallel_generatedRoot_36.sem (F.c0.h36_5 o)) sigmas8_odd hsm hg' hg he
      hhelpS hswitchS (hμS o) (hwalshS o)
    exact encRel_of_rel hr (fun a j ↦ by fin_cases a <;> fin_cases j <;> rfl)
      (fun j ↦ by fin_cases j <;> rfl)
  -- Placed at `X^o`.
  have hplace : ∀ o : Fin 8, EncRel sm (generatedRoot.w_37_0 ctx o, (generatedRoot.w_37_1 ctx o,
      (generatedRoot.w_37_2 ctx o, (generatedRoot.w_37_3 ctx o, (generatedRoot.w_37_4 ctx o,
        (generatedRoot.w_37_5 ctx o, ()))))))
      (fun j ↦ reducePoly modulus_0 8 (Xe (n := 8) o) •
        (reducePoly modulus_0 8 (Δ' * νs o * unitPair sp j) • gq))
      (7 * ((25 + 25) * 8 * Bh' * 2048 + (25 + 25) * 8 *
          (7 * ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048) +
            ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048) + 8 * (8 * 1 * BB)) * 2048) +
        ((25 + 25) * 8 * Bh' * 2048 + (25 + 25) * 8 *
          (7 * ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048) +
            ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048) + 8 * (8 * 1 * BB)) * 2048) +
        8 * (8 * 1 * BB')) := fun o ↦
    monomial_place_rel (by norm_num) (by have := o.isLt; omega)
      (parallel_generatedRoot_37.sem (F.c0.h37_5 o)) (hsamp o)
  -- Summed over the members.
  have ht := member_total_rel (M := 8) (by norm_num)
    (fams := (generatedRoot.w_37_0 ctx, (generatedRoot.w_37_1 ctx, (generatedRoot.w_37_2 ctx,
      (generatedRoot.w_37_3 ctx, (generatedRoot.w_37_4 ctx, (generatedRoot.w_37_5 ctx, ())))))))
    (a := (generatedRoot.w_38_0 ctx, (generatedRoot.w_41_0 ctx, (generatedRoot.w_44_0 ctx,
      (generatedRoot.w_47_0 ctx, (generatedRoot.w_50_0 ctx, (generatedRoot.w_53_0 ctx, ())))))))
    (b := (generatedRoot.w_39_0 ctx, (generatedRoot.w_42_0 ctx, (generatedRoot.w_45_0 ctx,
      (generatedRoot.w_48_0 ctx, (generatedRoot.w_51_0 ctx, (generatedRoot.w_54_0 ctx, ())))))))
    (fun c ↦ by
      fin_cases c
      exacts [F.c0.h38_0, F.c0.h41_0, F.c0.h44_0, F.c0.h47_0, F.c0.h50_0, F.c0.h53_0])
    (fun c ↦ by
      fin_cases c
      exacts [F.c0.h39_0, F.c0.h42_0, F.c0.h45_0, F.c0.h48_0, F.c0.h51_0, F.c0.h54_0])
    (fun c ↦ by fin_cases c <;> rfl) F.c0.h56_5
    (fun i cur next hb ↦ sequential_generatedRoot_56.sem hb) (fun o ↦ hplace o)
  exact ht.mono (le_of_eq rfl)

end Dio2Proof
