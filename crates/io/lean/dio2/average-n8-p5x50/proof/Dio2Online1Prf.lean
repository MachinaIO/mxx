import Dio2CellBridge
import Dio2LutBridge
import Gen_W_Stage_online_1_Scopes
import W_Stage_online_1_Facts
import GenInstLemmas

/-!
# The smudge PRF of step 1's online stage: the first bucket

Cell `i` of the first bucket (gather `parallel_generatedRoot_148`, 16 cells, two per member)
multiplies label member `t_i` (the gather table `w_139`) by the static state and by the selector
right-hand side of the cell; member `o` sums its two cells (`parallel_generatedRoot_149`). With
one-hot selectors, member `o` encodes `α_key μ` for its selected label member `key`
(`GenInst.bucket_of_cells`).
-/

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof.Gen RefreshTopProof.GenIface
  RefreshTopProof.GenInst W_Stage_online_1

/-- The tag prefix of the label root objects (`rns-bgg/root-input/dio2/label-0`). -/
abbrev labelPre : Blob :=
  [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100,
    105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]

/-- The tag prefix of the step-1 smudge PRF's selector tops (`rns-bgg/rhs/RhsId(0)/0/1`). -/
abbrev rhsPre1 : Blob :=
  [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 48, 41, 47,
    48, 47, 49]

/-- The parameter and Program identities in the step-1 selector tags. -/
abbrev prfA1 : Int := 36550556566445887949135852474979514307313305075533710697673106717124183540354
abbrev prfB1 : Int := 97925267724003222340535519505145373065489256943849140928644983866610518685471

set_option maxHeartbeats 16000000 in
/-- **The first bucket of the step-1 smudge PRF.** -/
theorem online1_bucket0 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {sm sp μ : ErrorPoly 8} {α : Fin 2 → ErrorPoly 8} (hα : ∀ k, polyNorm (α k) ≤ 1)
    (hαμ : ∀ k, polyNorm (α k * μ) ≤ 1)
    {G : ExactMatrix modulus_0 8 1 25} (hG : gadgetMatrixRuns backend 4096 25 G) {Ba : Nat}
    (ha : ∀ k : Fin 2, Rel sm (labelKeysG hashModel (generatedRoot.w_14_0 ctx) modulus_0 25 labelPre k)
      (fun j ↦ reducePoly modulus_0 8 (α k * unitPair sm j) • G)
      ![generatedRoot.w_16_0 ctx k, generatedRoot.w_27_0 ctx k] Ba)
    (hpay : ∀ k, generatedRoot.w_47_0 ctx k = fun _ _ ↦ reducePoly modulus_0 8 (α k))
    {g' g : ExactMatrix modulus_0 8 1 25} (hg' : gadgetMatrixRuns backend 4096 25 g')
    (hg : centeredRebaseRuns g' g) {Bs : Nat}
    (hs : Rel sm ![![generatedRoot.w_140_0 ctx, generatedRoot.w_143_0 ctx],
        ![generatedRoot.w_141_0 ctx, generatedRoot.w_144_0 ctx]]
      (fun j ↦ reducePoly modulus_0 8 (μ * unitPair sp j) • g)
      ![generatedRoot.w_142_0 ctx, generatedRoot.w_145_0 ctx] Bs)
    {ν : Fin 16 → ErrorPoly 8} {hT : ExactMatrix modulus_0 8 1 25}
    {eB : Fin 16 → Fin 2 → ErrorMatrix 8 1 25} {BB : Nat}
    (hrhs : ∀ (i : Fin 16) j, reducePoly modulus_0 8 sp •
        topPairG hashModel (generatedRoot.w_70_0 ctx) modulus_0 25 rhsPre1 (prfTopTags prfA1 prfB1 i) j +
        ![generatedRoot.w_146_0 ctx i, generatedRoot.w_147_0 ctx i] j =
      reducePoly modulus_0 8 (((1 : Nat) : Int) * ν i * unitPair sp j) • hT +
        reduceMatrix modulus_0 8 1 25 (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB)
    (sel : Fin 8 → Fin 2) (hsel : ∀ o j, ν (cellAt o j) = if j = sel o then (1 : ErrorPoly 8) else 0)
    (o : Fin 8) :
    ∃ key : Fin 2, ((key : Nat) : Int) = generatedRoot.w_139_0 ctx (cellAt o (sel o)) ∧
      EncRel sm (generatedRoot.w_149_0 ctx o, (generatedRoot.w_149_1 ctx o,
        (generatedRoot.w_149_2 ctx o, (generatedRoot.w_149_3 ctx o, (generatedRoot.w_149_4 ctx o,
          (generatedRoot.w_149_5 ctx o, ()))))))
        (fun i ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * (α key * μ) * unitPair sp i) • hT)
        (2 * ((25 + 25) * 8 * ((25 + 25) * 8 * Ba * 2048 + 8 * 1 * Bs) * 2048 + 8 * 1 * BB)) := by
  have hcell : ∀ (o : Fin 8) (j : Fin 2), ∃ k : Fin 2,
      ((k : Nat) : Int) = generatedRoot.w_139_0 ctx (cellAt o j) ∧
      EncRel sm (Tup.ev (k := 6) (generatedRoot.w_148_0 ctx, (generatedRoot.w_148_1 ctx,
          (generatedRoot.w_148_2 ctx, (generatedRoot.w_148_3 ctx, (generatedRoot.w_148_4 ctx,
            (generatedRoot.w_148_5 ctx, ())))))) (cellAt o j))
        (fun i ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * (α k * μ * ν (cellAt o j)) *
          unitPair sp i) • hT)
        ((25 + 25) * 8 * ((25 + 25) * 8 * Ba * 2048 + 8 * 1 * Bs) * 2048 + 8 * 1 * BB) := by
    intro o j
    obtain ⟨x0, x1, xp, -, -, g0, -, -, g1, -, -, g2, hc⟩ := F.c6.h148_5 (cellAt o j)
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
      (parallel_generatedRoot_148.sem hc) hG (hα k) (hαμ k) hak (hpay k) hg' hg hs
      (hrhs (cellAt o j)) (heB _)
    refine encRel_of_rel (hrel.target_eq ?_) (fun a j ↦ by fin_cases a <;> fin_cases j <;> rfl)
      (fun j ↦ by fin_cases j <;> rfl)
    funext i
    rw [smul_smul, ← map_mul]
    congr 2
    ring
  obtain ⟨key, hkey, -⟩ := hcell o (sel o)
  exact ⟨key, hkey, bucket_of_cells (M := 8) (W := 2) (K := 2) (p := 1) α (fun _ ↦ μ)
    (generatedRoot.w_139_0 ctx) ν hcell o (parallel_generatedRoot_149.sem (F.c6.h149_5 o))
    (sel o) (hsel o) key hkey⟩

/-- The tag prefix of the step-1 smudge PRF's second-bucket selector tops (`rns-bgg/rhs/RhsId(1)/1/2`). -/
abbrev rhsPre1b : Blob :=
  [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 49, 41, 47,
    49, 47, 50]

set_option maxHeartbeats 16000000 in
/-- **The second bucket of the step-1 smudge PRF.** Cell `i` multiplies label member `t'_i` (the
gather table `w_138`) by the state of member `⌊i / 2⌋` (the first bucket's output, `hstate`) and by
its selector right-hand side; member `o` encodes `α_key μ_o`. -/
theorem online1_bucket1 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {sm sp : ErrorPoly 8} {μ : Fin 8 → ErrorPoly 8} {α : Fin 2 → ErrorPoly 8}
    (hα : ∀ k, polyNorm (α k) ≤ 1) (hαμ : ∀ k o, polyNorm (α k * μ o) ≤ 1)
    {G : ExactMatrix modulus_0 8 1 25} (hG : gadgetMatrixRuns backend 4096 25 G) {Ba : Nat}
    (ha : ∀ k : Fin 2, Rel sm (labelKeysG hashModel (generatedRoot.w_14_0 ctx) modulus_0 25 labelPre k)
      (fun j ↦ reducePoly modulus_0 8 (α k * unitPair sm j) • G)
      ![generatedRoot.w_16_0 ctx k, generatedRoot.w_27_0 ctx k] Ba)
    (hpay : ∀ k, generatedRoot.w_47_0 ctx k = fun _ _ ↦ reducePoly modulus_0 8 (α k))
    {g' g : ExactMatrix modulus_0 8 1 25} (hg' : gadgetMatrixRuns backend 4096 25 g')
    (hg : centeredRebaseRuns g' g) {Bs : Nat}
    (hstate : ∀ o : Fin 8, EncRel sm (generatedRoot.w_149_0 ctx o, (generatedRoot.w_149_1 ctx o,
        (generatedRoot.w_149_2 ctx o, (generatedRoot.w_149_3 ctx o, (generatedRoot.w_149_4 ctx o,
          (generatedRoot.w_149_5 ctx o, ()))))))
      (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * μ o * unitPair sp j) • g) Bs)
    {ν : Fin 16 → ErrorPoly 8} {hT : ExactMatrix modulus_0 8 1 25}
    {eB : Fin 16 → Fin 2 → ErrorMatrix 8 1 25} {BB : Nat}
    (hrhs : ∀ (i : Fin 16) j, reducePoly modulus_0 8 sp •
        topPairG hashModel (generatedRoot.w_70_0 ctx) modulus_0 25 rhsPre1b (prfTopTags prfA1 prfB1 i) j +
        ![generatedRoot.w_150_0 ctx i, generatedRoot.w_151_0 ctx i] j =
      reducePoly modulus_0 8 (((1 : Nat) : Int) * ν i * unitPair sp j) • hT +
        reduceMatrix modulus_0 8 1 25 (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB)
    (sel : Fin 8 → Fin 2) (hsel : ∀ o j, ν (cellAt o j) = if j = sel o then (1 : ErrorPoly 8) else 0)
    (o : Fin 8) :
    ∃ key : Fin 2, ((key : Nat) : Int) = generatedRoot.w_138_0 ctx (cellAt o (sel o)) ∧
      EncRel sm (generatedRoot.w_153_0 ctx o, (generatedRoot.w_153_1 ctx o,
        (generatedRoot.w_153_2 ctx o, (generatedRoot.w_153_3 ctx o, (generatedRoot.w_153_4 ctx o,
          (generatedRoot.w_153_5 ctx o, ()))))))
        (fun i ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * (α key * μ o) * unitPair sp i) • hT)
        (2 * ((25 + 25) * 8 * ((25 + 25) * 8 * Ba * 2048 + 8 * 1 * Bs) * 2048 + 8 * 1 * BB)) := by
  have hcell : ∀ (o : Fin 8) (j : Fin 2), ∃ k : Fin 2,
      ((k : Nat) : Int) = generatedRoot.w_138_0 ctx (cellAt o j) ∧
      EncRel sm (Tup.ev (k := 6) (generatedRoot.w_152_0 ctx, (generatedRoot.w_152_1 ctx,
          (generatedRoot.w_152_2 ctx, (generatedRoot.w_152_3 ctx, (generatedRoot.w_152_4 ctx,
            (generatedRoot.w_152_5 ctx, ())))))) (cellAt o j))
        (fun i ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * (α k * μ o * ν (cellAt o j)) *
          unitPair sp i) • hT)
        ((25 + 25) * 8 * ((25 + 25) * 8 * Ba * 2048 + 8 * 1 * Bs) * 2048 + 8 * 1 * BB) := by
    intro o j
    obtain ⟨x0, x1, xp, -, -, g0, -, -, g1, -, -, g2, hc⟩ := F.c6.h152_5 (cellAt o j)
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
      (parallel_generatedRoot_152.sem hc) hio hG (hα k) (by rw [hoo]; exact hαμ k o) hak (hpay k)
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
    (generatedRoot.w_138_0 ctx) ν hcell o (parallel_generatedRoot_153.sem (F.c6.h153_5 o))
    (sel o) (hsel o) key hkey⟩

/-- The tag prefix of the LUT helpers' root objects (`rns-bgg/root-lut-a`). -/
abbrev helperPre : Blob := [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]

/-- The automorphisms of a LUT of width eight. -/
abbrev sigmas8 : List Int := [1, 3, 5, 7, 9, 11, 13, 15]

theorem sigmas8_odd : ∀ i : Fin 8, Odd (sigmaOf sigmas8 i).toNat := by
  decide

set_option maxHeartbeats 32000000 in
/-- **The step-1 smudge PRF's readout LUT, placed and summed.** Member `o` of the second bucket
encodes `μ_o` (`hin`); its readout LUT (`parallel_generatedRoot_154`) encodes `Δ' ν_o`
(`LutMemberFlat.rel`, the LUT identity `hwalsh o`), placed at `X^o` (`parallel_generatedRoot_155`)
and summed over the eight members (`sequential_generatedRoot_174`). -/
theorem online1_smudge_total {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {sm sp : ErrorPoly 8} (hsm : polyNorm sm ≤ 1)
    {g' g : ExactMatrix modulus_0 8 1 25} (hg' : gadgetMatrixRuns backend 4096 25 g')
    (hg : centeredRebaseRuns g' g) {μ : Fin 8 → ErrorPoly 8} {B : Nat}
    (hin : ∀ o : Fin 8, EncRel sm (generatedRoot.w_153_0 ctx o, (generatedRoot.w_153_1 ctx o,
        (generatedRoot.w_153_2 ctx o, (generatedRoot.w_153_3 ctx o, (generatedRoot.w_153_4 ctx o,
          (generatedRoot.w_153_5 ctx o, ()))))))
      (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * μ o * unitPair sp j) • g) B)
    {Bh : Nat}
    (hhelp : ∀ i : Fin 8, Rel sm (keyPairG hashModel (generatedRoot.w_14_0 ctx) modulus_0 25 helperPre
        [.integer (sigmaOf sigmas8 i)])
      (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) *
        unitPair (substitution (sigmaOf sigmas8 i).toNat (sigmas8_odd i) sm) j) • g)
      ![generatedRoot.w_136_0 ctx i, generatedRoot.w_137_0 ctx i] Bh)
    {Δ' : ErrorPoly 8} {ν : Fin 8 → ErrorPoly 8} {gq : ExactMatrix modulus_0 8 1 25}
    {C : Fin 8 → ExactPoly modulus_0 8} {BB : Nat}
    (hswitch : ∀ i : Fin 8, ∃ e : Fin 2 → ErrorMatrix 8 1 25, (∀ j, CoeffBound (e j) BB) ∧ ∀ j,
      reducePoly modulus_0 8 (substitution (sigmaOf sigmas8 i).toNat (sigmas8_odd i) sp) •
          ![![generatedRoot.w_132_0 ctx, generatedRoot.w_133_0 ctx],
            ![generatedRoot.w_134_0 ctx, generatedRoot.w_135_0 ctx]] 0 j i +
        ![![generatedRoot.w_132_0 ctx, generatedRoot.w_133_0 ctx],
          ![generatedRoot.w_134_0 ctx, generatedRoot.w_135_0 ctx]] 1 j i =
      (C i * reducePoly modulus_0 8 (Δ' * unitPair sp j)) • gq + reduceMatrix modulus_0 8 1 25 (e j))
    (hμ : ∀ o i, polyNorm (substitution (sigmaOf sigmas8 i).toNat (sigmas8_odd i) (μ o)) ≤ 1)
    (hwalsh : ∀ o, ∑ i, reducePoly modulus_0 8 (substitution (sigmaOf sigmas8 i).toNat
      (sigmas8_odd i) (μ o)) * C i = reducePoly modulus_0 8 (ν o)) :
    EncRel sm (generatedRoot.w_174_0 ctx, (generatedRoot.w_174_1 ctx, (generatedRoot.w_174_2 ctx,
        (generatedRoot.w_174_3 ctx, (generatedRoot.w_174_4 ctx, (generatedRoot.w_174_5 ctx, ()))))))
      (∑ o : Fin 8, fun j ↦ reducePoly modulus_0 8 (Xe (n := 8) o) •
        (reducePoly modulus_0 8 (Δ' * ν o * unitPair sp j) • gq))
      (8 * (7 * ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048) +
        ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048) + 8 * (8 * 1 * BB))) := by
  -- Every member's readout LUT.
  have hmem : ∀ o : Fin 8, EncRel sm (generatedRoot.w_154_0 ctx o, (generatedRoot.w_154_1 ctx o,
      (generatedRoot.w_154_2 ctx o, (generatedRoot.w_154_3 ctx o, (generatedRoot.w_154_4 ctx o,
        (generatedRoot.w_154_5 ctx o, ()))))))
      (fun j ↦ reducePoly modulus_0 8 (Δ' * ν o * unitPair sp j) • gq)
      (7 * ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048) +
        ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048) + 8 * (8 * 1 * BB)) := by
    intro o
    have he := hin o
    unfold EncRel at he
    have hr := LutMemberFlat.rel (by norm_num) (log := 3) (by norm_num) (L := 7) (by norm_num) hq1
      layout hlayout hexact (fun i a b hb ↦ sequential_parallel_generatedRoot_154_18.sem hb)
      (parallel_generatedRoot_154.sem (F.c6.h154_5 o)) sigmas8_odd hsm hg' hg he
      hhelp hswitch (hμ o) (hwalsh o)
    exact encRel_of_rel hr (fun a j ↦ by fin_cases a <;> fin_cases j <;> rfl)
      (fun j ↦ by fin_cases j <;> rfl)
  -- Placed at `X^o`.
  have hplace : ∀ o : Fin 8, EncRel sm (generatedRoot.w_155_0 ctx o, (generatedRoot.w_155_1 ctx o,
      (generatedRoot.w_155_2 ctx o, (generatedRoot.w_155_3 ctx o, (generatedRoot.w_155_4 ctx o,
        (generatedRoot.w_155_5 ctx o, ()))))))
      (fun j ↦ reducePoly modulus_0 8 (Xe (n := 8) o) •
        (reducePoly modulus_0 8 (Δ' * ν o * unitPair sp j) • gq))
      (7 * ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048) +
        ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048) + 8 * (8 * 1 * BB)) := fun o ↦
    place_rel (by norm_num) (parallel_generatedRoot_155.sem (F.c6.h155_5 o)) (hmem o)
      (by rw [Int.fmod_eq_emod_of_nonneg _ (by norm_num)]; have := o.isLt; omega) (by omega)
  -- Summed over the members.
  have ht := member_total_rel (M := 8) (by norm_num)
    (fams := (generatedRoot.w_155_0 ctx, (generatedRoot.w_155_1 ctx, (generatedRoot.w_155_2 ctx,
      (generatedRoot.w_155_3 ctx, (generatedRoot.w_155_4 ctx, (generatedRoot.w_155_5 ctx, ())))))))
    (a := (generatedRoot.w_156_0 ctx, (generatedRoot.w_159_0 ctx, (generatedRoot.w_162_0 ctx,
      (generatedRoot.w_165_0 ctx, (generatedRoot.w_168_0 ctx, (generatedRoot.w_171_0 ctx, ())))))))
    (b := (generatedRoot.w_157_0 ctx, (generatedRoot.w_160_0 ctx, (generatedRoot.w_163_0 ctx,
      (generatedRoot.w_166_0 ctx, (generatedRoot.w_169_0 ctx, (generatedRoot.w_172_0 ctx, ())))))))
    (fun c ↦ by
      fin_cases c
      exacts [F.c6.h156_0, F.c6.h159_0, F.c6.h162_0, F.c6.h165_0, F.c7.h168_0, F.c7.h171_0])
    (fun c ↦ by
      fin_cases c
      exacts [F.c6.h157_0, F.c6.h160_0, F.c6.h163_0, F.c7.h166_0, F.c7.h169_0, F.c7.h172_0])
    (fun c ↦ by fin_cases c <;> rfl) F.c7.h174_5
    (fun i cur next hb ↦ sequential_generatedRoot_174.sem hb) (fun o ↦ hplace o)
  exact ht.mono (le_of_eq rfl)

end Dio2Proof
