import Dio2Online0Sigma
import Gen_W_Stage_online_0_Scopes
import W_Stage_online_0_Facts

/-!
# The fresh-noise and smudge PRFs of step 0's online stage

Both have the shape of step 1's smudge PRF (`Dio2Online1Prf`) with many members, followed by a
`Step::Sum` row of columns instead of a single member total:

* **CBD fresh noise** (7200 members, 14400 cells): the first bucket (static cells
  `parallel_generatedRoot_1181`, row sums `parallel_generatedRoot_1182`), the second bucket
  (dynamic cells `parallel_generatedRoot_1185`, row sums `parallel_generatedRoot_1186`), per member
  the readout LUT (`parallel_generatedRoot_1187`) placed at `X^(o mod 8)`
  (`parallel_generatedRoot_1188`, role `sampler.placement`), then column `0` (member loop
  `sequential_generatedRoot_1207`) and the column loop over the 900 columns of eight members
  (`sequential_generatedRoot_1219`).
* **Smudge** (36000 members, 72000 cells): cells `1235`/`1239`, row sums `1236`/`1240`, LUT
  `1241`, placement `1242`, member loop `1261`, column loop `1273` (4500 columns).

The row results are stated as refresh_top's `GenInst.column_total_rel` states them: block `cc` of
the row encodes the sum of the placed targets of members `8 cc + j`, `j < 8`.
-/

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof.Gen RefreshTopProof.GenIface
  RefreshTopProof.GenInst W_Stage_online_0

/-- The tag prefix of the step-0 fresh-noise PRF's first-bucket selector tops
(`rns-bgg/rhs/RhsId(5)/0/1`). -/
abbrev rhsPreF0 : Blob :=
  [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 53, 41, 47,
    48, 47, 49]

/-- The tag prefix of the step-0 fresh-noise PRF's second-bucket selector tops
(`rns-bgg/rhs/RhsId(6)/1/2`). -/
abbrev rhsPreF0b : Blob :=
  [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 54, 41, 47,
    49, 47, 50]

/-! The step-0 smudge PRF's selector tops use `rhsPre1` (`RhsId(0)/0/1`) and `rhsPre1b`
(`RhsId(1)/1/2`) of `Dio2Online1Prf`, with the step-0 identities `prfA0`, `prfB0`. -/

set_option maxHeartbeats 16000000 in
/-- **The first bucket of the step-0 fresh-noise PRF.** Cell `i` multiplies label member `t_i` (the
gather table `w_1171`) by the static state and by its selector right-hand side; member `o` sums its
two cells. -/
theorem online0_fresh_bucket0 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {sm sp μ : ErrorPoly 8} {α : Fin 2 → ErrorPoly 8} (hα : ∀ k, polyNorm (α k) ≤ 1)
    (hαμ : ∀ k, polyNorm (α k * μ) ≤ 1)
    {G : ExactMatrix modulus_0 8 1 25} (hG : gadgetMatrixRuns backend 4096 25 G) {Ba : Nat}
    (ha : ∀ k : Fin 2, Rel sm (labelKeysG hashModel (generatedRoot.w_7_0 ctx) modulus_0 25 labelPre k)
      (fun j ↦ reducePoly modulus_0 8 (α k * unitPair sm j) • G)
      ![generatedRoot.w_1169_0 ctx k, generatedRoot.w_1170_0 ctx k] Ba)
    (hpay : ∀ k, generatedRoot.w_1175_0 ctx k = fun _ _ ↦ reducePoly modulus_0 8 (α k))
    {g' g : ExactMatrix modulus_0 8 1 25} (hg' : gadgetMatrixRuns backend 4096 25 g')
    (hg : centeredRebaseRuns g' g) {Bs : Nat}
    (hs : Rel sm ![![generatedRoot.w_1172_0 ctx, generatedRoot.w_1176_0 ctx],
        ![generatedRoot.w_1173_0 ctx, generatedRoot.w_1177_0 ctx]]
      (fun j ↦ reducePoly modulus_0 8 (μ * unitPair sp j) • g)
      ![generatedRoot.w_1174_0 ctx, generatedRoot.w_1178_0 ctx] Bs)
    {ν : Fin 14400 → ErrorPoly 8} {hT : ExactMatrix modulus_0 8 1 25}
    {eB : Fin 14400 → Fin 2 → ErrorMatrix 8 1 25} {BB : Nat}
    (hrhs : ∀ (i : Fin 14400) j, reducePoly modulus_0 8 sp •
        topPairG hashModel (generatedRoot.w_26_0 ctx) modulus_0 25 rhsPreF0 (prfTopTags prfA0 prfB0 i) j +
        ![generatedRoot.w_1179_0 ctx i, generatedRoot.w_1180_0 ctx i] j =
      reducePoly modulus_0 8 (((1 : Nat) : Int) * ν i * unitPair sp j) • hT +
        reduceMatrix modulus_0 8 1 25 (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB)
    (sel : Fin 7200 → Fin 2)
    (hsel : ∀ o j, ν (cellAt o j) = if j = sel o then (1 : ErrorPoly 8) else 0)
    (o : Fin 7200) :
    ∃ key : Fin 2, ((key : Nat) : Int) = generatedRoot.w_1171_0 ctx (cellAt o (sel o)) ∧
      EncRel sm (generatedRoot.w_1182_0 ctx o, (generatedRoot.w_1182_1 ctx o, (generatedRoot.w_1182_2 ctx o, (generatedRoot.w_1182_3 ctx o, (generatedRoot.w_1182_4 ctx o, (generatedRoot.w_1182_5 ctx o, ()))))))
        (fun i ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * (α key * μ) * unitPair sp i) • hT)
        (2 * ((25 + 25) * 8 * ((25 + 25) * 8 * Ba * 2048 + 8 * 1 * Bs) * 2048 + 8 * 1 * BB)) := by
  have hcell : ∀ (o : Fin 7200) (j : Fin 2), ∃ k : Fin 2,
      ((k : Nat) : Int) = generatedRoot.w_1171_0 ctx (cellAt o j) ∧
      EncRel sm (Tup.ev (k := 6) (generatedRoot.w_1181_0 ctx, (generatedRoot.w_1181_1 ctx, (generatedRoot.w_1181_2 ctx, (generatedRoot.w_1181_3 ctx, (generatedRoot.w_1181_4 ctx, (generatedRoot.w_1181_5 ctx, ())))))) (cellAt o j))
        (fun i ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * (α k * μ * ν (cellAt o j)) *
          unitPair sp i) • hT)
        ((25 + 25) * 8 * ((25 + 25) * 8 * Ba * 2048 + 8 * 1 * Bs) * 2048 + 8 * 1 * BB) := by
    intro o j
    obtain ⟨x0, x1, xp, -, -, g0, -, -, g1, -, -, g2, hc⟩ := F.c45.h1181_5 (cellAt o j)
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
      (parallel_generatedRoot_1181.sem hc) hG (hα k) (hαμ k) hak (hpay k) hg' hg hs
      (hrhs (cellAt o j)) (heB _)
    refine encRel_of_rel (hrel.target_eq ?_) (fun a j ↦ by fin_cases a <;> fin_cases j <;> rfl)
      (fun j ↦ by fin_cases j <;> rfl)
    funext i
    rw [smul_smul, ← map_mul]
    congr 2
    ring
  obtain ⟨key, hkey, -⟩ := hcell o (sel o)
  exact ⟨key, hkey, bucket_of_cells (M := 7200) (W := 2) (K := 2) (p := 1) α (fun _ ↦ μ)
    (generatedRoot.w_1171_0 ctx) ν hcell o (parallel_generatedRoot_1182.sem (F.c45.h1182_5 o))
    (sel o) (hsel o) key hkey⟩

set_option maxHeartbeats 16000000 in
/-- **The second bucket of the step-0 fresh-noise PRF.** Cell `i` multiplies label member `t'_i`
(the gather table `w_1168`) by the state of member `⌊i / 2⌋` (`hstate`) and by its selector
right-hand side; member `o` encodes `α_key μ_o`. -/
theorem online0_fresh_bucket1 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {sm sp : ErrorPoly 8} {μ : Fin 7200 → ErrorPoly 8} {α : Fin 2 → ErrorPoly 8}
    (hα : ∀ k, polyNorm (α k) ≤ 1) (hαμ : ∀ k o, polyNorm (α k * μ o) ≤ 1)
    {G : ExactMatrix modulus_0 8 1 25} (hG : gadgetMatrixRuns backend 4096 25 G) {Ba : Nat}
    (ha : ∀ k : Fin 2, Rel sm (labelKeysG hashModel (generatedRoot.w_7_0 ctx) modulus_0 25 labelPre k)
      (fun j ↦ reducePoly modulus_0 8 (α k * unitPair sm j) • G)
      ![generatedRoot.w_1169_0 ctx k, generatedRoot.w_1170_0 ctx k] Ba)
    (hpay : ∀ k, generatedRoot.w_1175_0 ctx k = fun _ _ ↦ reducePoly modulus_0 8 (α k))
    {g' g : ExactMatrix modulus_0 8 1 25} (hg' : gadgetMatrixRuns backend 4096 25 g')
    (hg : centeredRebaseRuns g' g) {Bs : Nat}
    (hstate : ∀ o : Fin 7200, EncRel sm (generatedRoot.w_1182_0 ctx o, (generatedRoot.w_1182_1 ctx o, (generatedRoot.w_1182_2 ctx o, (generatedRoot.w_1182_3 ctx o, (generatedRoot.w_1182_4 ctx o, (generatedRoot.w_1182_5 ctx o, ()))))))
      (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * μ o * unitPair sp j) • g) Bs)
    {ν : Fin 14400 → ErrorPoly 8} {hT : ExactMatrix modulus_0 8 1 25}
    {eB : Fin 14400 → Fin 2 → ErrorMatrix 8 1 25} {BB : Nat}
    (hrhs : ∀ (i : Fin 14400) j, reducePoly modulus_0 8 sp •
        topPairG hashModel (generatedRoot.w_26_0 ctx) modulus_0 25 rhsPreF0b (prfTopTags prfA0 prfB0 i) j +
        ![generatedRoot.w_1183_0 ctx i, generatedRoot.w_1184_0 ctx i] j =
      reducePoly modulus_0 8 (((1 : Nat) : Int) * ν i * unitPair sp j) • hT +
        reduceMatrix modulus_0 8 1 25 (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB)
    (sel : Fin 7200 → Fin 2)
    (hsel : ∀ o j, ν (cellAt o j) = if j = sel o then (1 : ErrorPoly 8) else 0)
    (o : Fin 7200) :
    ∃ key : Fin 2, ((key : Nat) : Int) = generatedRoot.w_1168_0 ctx (cellAt o (sel o)) ∧
      EncRel sm (generatedRoot.w_1186_0 ctx o, (generatedRoot.w_1186_1 ctx o, (generatedRoot.w_1186_2 ctx o, (generatedRoot.w_1186_3 ctx o, (generatedRoot.w_1186_4 ctx o, (generatedRoot.w_1186_5 ctx o, ()))))))
        (fun i ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * (α key * μ o) * unitPair sp i) • hT)
        (2 * ((25 + 25) * 8 * ((25 + 25) * 8 * Ba * 2048 + 8 * 1 * Bs) * 2048 + 8 * 1 * BB)) := by
  have hcell : ∀ (o : Fin 7200) (j : Fin 2), ∃ k : Fin 2,
      ((k : Nat) : Int) = generatedRoot.w_1168_0 ctx (cellAt o j) ∧
      EncRel sm (Tup.ev (k := 6) (generatedRoot.w_1185_0 ctx, (generatedRoot.w_1185_1 ctx, (generatedRoot.w_1185_2 ctx, (generatedRoot.w_1185_3 ctx, (generatedRoot.w_1185_4 ctx, (generatedRoot.w_1185_5 ctx, ())))))) (cellAt o j))
        (fun i ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * (α k * μ o * ν (cellAt o j)) *
          unitPair sp i) • hT)
        ((25 + 25) * 8 * ((25 + 25) * 8 * Ba * 2048 + 8 * 1 * Bs) * 2048 + 8 * 1 * BB) := by
    intro o j
    obtain ⟨x0, x1, xp, -, -, g0, -, -, g1, -, -, g2, hc⟩ := F.c45.h1185_5 (cellAt o j)
    obtain ⟨k, hk, rfl⟩ := g0
    obtain ⟨k1, hk1, rfl⟩ := g1
    obtain ⟨k2, hk2, rfl⟩ := g2
    have e1 : k = k1 := Fin.ext (by omega)
    have e2 : k = k2 := Fin.ext (by omega)
    subst e1 e2
    refine ⟨k, hk, ?_⟩
    have hak := ha k
    rw [hk] at hak
    have hio : (cellAt o j : Nat) / 2 < 7200 := by
      have := o.isLt; have := j.isLt; simp only [cellAt]; omega
    have hoo : (⟨(cellAt o j : Nat) / 2, hio⟩ : Fin 7200) = o := by
      apply Fin.ext; have := j.isLt; simp only [cellAt]; omega
    have hs := hstate o
    rw [← hoo] at hs
    unfold EncRel at hs
    have hrel := PrfCellFlat.rel_1 (by norm_num) hq1 layout hlayout hexact
      (parallel_generatedRoot_1185.sem hc) hio hG (hα k) (by rw [hoo]; exact hαμ k o) hak (hpay k)
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
  exact ⟨key, hkey, bucket_of_cells (M := 7200) (W := 2) (K := 2) (p := 1) α μ
    (generatedRoot.w_1168_0 ctx) ν hcell o (parallel_generatedRoot_1186.sem (F.c45.h1186_5 o))
    (sel o) (hsel o) key hkey⟩

set_option maxHeartbeats 32000000 in
/-- **The step-0 fresh-noise row.** Member `o` of the second bucket encodes `μ_o` (`hin`); its
readout LUT (`parallel_generatedRoot_1187`) encodes `Δ' ν_o` (LUT identity `hwalsh o`), placed at
`X^(o mod 8)` (`parallel_generatedRoot_1188`); block `cc` of the row
(`sequential_generatedRoot_1219`, column `0` from `sequential_generatedRoot_1207`) encodes the sum
of the placed members `8 cc + j`. -/
theorem online0_fresh_row {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {sm sp : ErrorPoly 8} (hsm : polyNorm sm ≤ 1)
    {g' g : ExactMatrix modulus_0 8 1 25} (hg' : gadgetMatrixRuns backend 4096 25 g')
    (hg : centeredRebaseRuns g' g) {μ : Fin 7200 → ErrorPoly 8} {B : Nat}
    (hin : ∀ o : Fin 7200, EncRel sm (generatedRoot.w_1186_0 ctx o, (generatedRoot.w_1186_1 ctx o, (generatedRoot.w_1186_2 ctx o, (generatedRoot.w_1186_3 ctx o, (generatedRoot.w_1186_4 ctx o, (generatedRoot.w_1186_5 ctx o, ()))))))
      (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * μ o * unitPair sp j) • g) B)
    {Bh : Nat}
    (hhelp : ∀ i : Fin 8, Rel sm (keyPairG hashModel (generatedRoot.w_7_0 ctx) modulus_0 25 helperPre
        [.integer (sigmaOf sigmas8 i)])
      (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) *
        unitPair (substitution (sigmaOf sigmas8 i).toNat (sigmas8_odd i) sm) j) • g)
      ![generatedRoot.w_13_0 ctx i, generatedRoot.w_14_0 ctx i] Bh)
    {Δ' : ErrorPoly 8} {ν : Fin 7200 → ErrorPoly 8} {gq : ExactMatrix modulus_0 8 1 25}
    {C : Fin 8 → ExactPoly modulus_0 8} {BB : Nat}
    (hswitch : ∀ i : Fin 8, ∃ e : Fin 2 → ErrorMatrix 8 1 25, (∀ j, CoeffBound (e j) BB) ∧ ∀ j,
      reducePoly modulus_0 8 (substitution (sigmaOf sigmas8 i).toNat (sigmas8_odd i) sp) •
          ![![generatedRoot.w_1164_0 ctx, generatedRoot.w_1165_0 ctx],
            ![generatedRoot.w_1166_0 ctx, generatedRoot.w_1167_0 ctx]] 0 j i +
        ![![generatedRoot.w_1164_0 ctx, generatedRoot.w_1165_0 ctx],
          ![generatedRoot.w_1166_0 ctx, generatedRoot.w_1167_0 ctx]] 1 j i =
      (C i * reducePoly modulus_0 8 (Δ' * unitPair sp j)) • gq + reduceMatrix modulus_0 8 1 25 (e j))
    (hμ : ∀ o i, polyNorm (substitution (sigmaOf sigmas8 i).toNat (sigmas8_odd i) (μ o)) ≤ 1)
    (hwalsh : ∀ o, ∑ i, reducePoly modulus_0 8 (substitution (sigmaOf sigmas8 i).toNat
      (sigmas8_odd i) (μ o)) * C i = reducePoly modulus_0 8 (ν o))
    (cc : Fin 900) :
    EncRel sm (blkT (r := 1 * 1) (C := 900) (m := 25) (generatedRoot.w_1219_0 ctx, (generatedRoot.w_1219_1 ctx, (generatedRoot.w_1219_2 ctx, (generatedRoot.w_1219_3 ctx, (generatedRoot.w_1219_4 ctx, (generatedRoot.w_1219_5 ctx, ())))))) cc)
      (∑ j : Fin 8, (fun (f : Fin 7200) (jj : Fin 2) ↦ reducePoly modulus_0 8 (Xe (n := 8) (f.val % 8)) •
          (reducePoly modulus_0 8 (Δ' * ν f * unitPair sp jj) • gq))
        ⟨8 * cc + j, by have := cc.isLt; have := j.isLt; omega⟩)
      (8 * (7 * ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048) +
        ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048) + 8 * (8 * 1 * BB))) := by
  -- Every member's readout LUT.
  have hmem : ∀ o : Fin 7200, EncRel sm (generatedRoot.w_1187_0 ctx o, (generatedRoot.w_1187_1 ctx o, (generatedRoot.w_1187_2 ctx o, (generatedRoot.w_1187_3 ctx o, (generatedRoot.w_1187_4 ctx o, (generatedRoot.w_1187_5 ctx o, ()))))))
      (fun j ↦ reducePoly modulus_0 8 (Δ' * ν o * unitPair sp j) • gq)
      (7 * ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048) +
        ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048) + 8 * (8 * 1 * BB)) := by
    intro o
    have he := hin o
    unfold EncRel at he
    have hr := LutMemberFlat.rel (by norm_num) (log := 3) (by norm_num) (L := 7) (by norm_num) hq1
      layout hlayout hexact (fun i a b hb ↦ sequential_parallel_generatedRoot_1187_18.sem hb)
      (parallel_generatedRoot_1187.sem (F.c45.h1187_5 o)) sigmas8_odd hsm hg' hg he
      hhelp hswitch (hμ o) (hwalsh o)
    exact encRel_of_rel hr (fun a j ↦ by fin_cases a <;> fin_cases j <;> rfl)
      (fun j ↦ by fin_cases j <;> rfl)
  -- Placed at `X^(o mod 8)`.
  have hplace : ∀ o : Fin 7200, EncRel sm (generatedRoot.w_1188_0 ctx o, (generatedRoot.w_1188_1 ctx o, (generatedRoot.w_1188_2 ctx o, (generatedRoot.w_1188_3 ctx o, (generatedRoot.w_1188_4 ctx o, (generatedRoot.w_1188_5 ctx o, ()))))))
      (fun j ↦ reducePoly modulus_0 8 (Xe (n := 8) (o.val % 8)) •
        (reducePoly modulus_0 8 (Δ' * ν o * unitPair sp j) • gq))
      (7 * ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048) +
        ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048) + 8 * (8 * 1 * BB)) := fun o ↦
    place_rel (by norm_num) (parallel_generatedRoot_1188.sem (F.c45.h1188_5 o)) (hmem o)
      (by rw [Int.fmod_eq_emod_of_nonneg _ (by norm_num)]; omega) (by omega)
  -- Column `0`: the members `0 … 7`.
  have hx0 := member_prefix (α := ExactMatrix modulus_0 8 1 25) (k := 6) (F := 7200) (Wm := 8) (by norm_num) (by norm_num)
    (fams := (generatedRoot.w_1188_0 ctx, (generatedRoot.w_1188_1 ctx, (generatedRoot.w_1188_2 ctx, (generatedRoot.w_1188_3 ctx, (generatedRoot.w_1188_4 ctx, (generatedRoot.w_1188_5 ctx, ())))))))
    (a := (generatedRoot.w_1189_0 ctx, (generatedRoot.w_1192_0 ctx, (generatedRoot.w_1195_0 ctx, (generatedRoot.w_1198_0 ctx, (generatedRoot.w_1201_0 ctx, (generatedRoot.w_1204_0 ctx, ())))))))
    (b := (generatedRoot.w_1190_0 ctx, (generatedRoot.w_1193_0 ctx, (generatedRoot.w_1196_0 ctx, (generatedRoot.w_1199_0 ctx, (generatedRoot.w_1202_0 ctx, (generatedRoot.w_1205_0 ctx, ())))))))
    (fun c ↦ by
      fin_cases c
      exacts [F.c45.h1189_0, F.c45.h1192_0, F.c45.h1195_0, F.c45.h1198_0, F.c45.h1201_0, F.c45.h1204_0])
    (fun c ↦ by
      fin_cases c
      exacts [F.c45.h1190_0, F.c45.h1193_0, F.c45.h1196_0, F.c45.h1199_0, F.c45.h1202_0, F.c46.h1205_0])
    (fun c ↦ by fin_cases c <;> rfl) F.c46.h1207_5
    (fun i cur next hb ↦ sequential_generatedRoot_1207.sem hb)
  -- The column loop.
  exact column_total_rel (C := 900) (m := 25) (W := 8) (F := 7200) (by norm_num) (by norm_num) hx0
    (fun c ↦ by
      fin_cases c
      exacts [F.c46.h1208_0, F.c46.h1210_0, F.c46.h1212_0, F.c46.h1214_0, F.c46.h1216_0, F.c46.h1218_0])
    F.c46.h1219_5
    -- Elaborated before unifying with  (otherwise the unifier unfolds the spec).
    (fun i cur next hb ↦ by have hc := sequential_generatedRoot_1219.sem hb; exact hc)
    (t := fun (f : Fin 7200) (jj : Fin 2) ↦ reducePoly modulus_0 8 (Xe (n := 8) (f.val % 8)) •
      (reducePoly modulus_0 8 (Δ' * ν f * unitPair sp jj) • gq))
    (fun o ↦ hplace o) cc

set_option maxHeartbeats 16000000 in
/-- **The first bucket of the step-0 smudge PRF** (gather table `w_1226`, label families
`w_257`/`w_262`, payloads `w_276`). -/
theorem online0_smudge_bucket0 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {sm sp μ : ErrorPoly 8} {α : Fin 2 → ErrorPoly 8} (hα : ∀ k, polyNorm (α k) ≤ 1)
    (hαμ : ∀ k, polyNorm (α k * μ) ≤ 1)
    {G : ExactMatrix modulus_0 8 1 25} (hG : gadgetMatrixRuns backend 4096 25 G) {Ba : Nat}
    (ha : ∀ k : Fin 2, Rel sm (labelKeysG hashModel (generatedRoot.w_7_0 ctx) modulus_0 25 labelPre k)
      (fun j ↦ reducePoly modulus_0 8 (α k * unitPair sm j) • G)
      ![generatedRoot.w_257_0 ctx k, generatedRoot.w_262_0 ctx k] Ba)
    (hpay : ∀ k, generatedRoot.w_276_0 ctx k = fun _ _ ↦ reducePoly modulus_0 8 (α k))
    {g' g : ExactMatrix modulus_0 8 1 25} (hg' : gadgetMatrixRuns backend 4096 25 g')
    (hg : centeredRebaseRuns g' g) {Bs : Nat}
    (hs : Rel sm ![![generatedRoot.w_1227_0 ctx, generatedRoot.w_1230_0 ctx],
        ![generatedRoot.w_1228_0 ctx, generatedRoot.w_1231_0 ctx]]
      (fun j ↦ reducePoly modulus_0 8 (μ * unitPair sp j) • g)
      ![generatedRoot.w_1229_0 ctx, generatedRoot.w_1232_0 ctx] Bs)
    {ν : Fin 72000 → ErrorPoly 8} {hT : ExactMatrix modulus_0 8 1 25}
    {eB : Fin 72000 → Fin 2 → ErrorMatrix 8 1 25} {BB : Nat}
    (hrhs : ∀ (i : Fin 72000) j, reducePoly modulus_0 8 sp •
        topPairG hashModel (generatedRoot.w_26_0 ctx) modulus_0 25 rhsPre1 (prfTopTags prfA0 prfB0 i) j +
        ![generatedRoot.w_1233_0 ctx i, generatedRoot.w_1234_0 ctx i] j =
      reducePoly modulus_0 8 (((1 : Nat) : Int) * ν i * unitPair sp j) • hT +
        reduceMatrix modulus_0 8 1 25 (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB)
    (sel : Fin 36000 → Fin 2)
    (hsel : ∀ o j, ν (cellAt o j) = if j = sel o then (1 : ErrorPoly 8) else 0)
    (o : Fin 36000) :
    ∃ key : Fin 2, ((key : Nat) : Int) = generatedRoot.w_1226_0 ctx (cellAt o (sel o)) ∧
      EncRel sm (generatedRoot.w_1236_0 ctx o, (generatedRoot.w_1236_1 ctx o, (generatedRoot.w_1236_2 ctx o, (generatedRoot.w_1236_3 ctx o, (generatedRoot.w_1236_4 ctx o, (generatedRoot.w_1236_5 ctx o, ()))))))
        (fun i ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * (α key * μ) * unitPair sp i) • hT)
        (2 * ((25 + 25) * 8 * ((25 + 25) * 8 * Ba * 2048 + 8 * 1 * Bs) * 2048 + 8 * 1 * BB)) := by
  have hcell : ∀ (o : Fin 36000) (j : Fin 2), ∃ k : Fin 2,
      ((k : Nat) : Int) = generatedRoot.w_1226_0 ctx (cellAt o j) ∧
      EncRel sm (Tup.ev (k := 6) (generatedRoot.w_1235_0 ctx, (generatedRoot.w_1235_1 ctx, (generatedRoot.w_1235_2 ctx, (generatedRoot.w_1235_3 ctx, (generatedRoot.w_1235_4 ctx, (generatedRoot.w_1235_5 ctx, ())))))) (cellAt o j))
        (fun i ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * (α k * μ * ν (cellAt o j)) *
          unitPair sp i) • hT)
        ((25 + 25) * 8 * ((25 + 25) * 8 * Ba * 2048 + 8 * 1 * Bs) * 2048 + 8 * 1 * BB) := by
    intro o j
    obtain ⟨x0, x1, xp, -, -, g0, -, -, g1, -, -, g2, hc⟩ := F.c46.h1235_5 (cellAt o j)
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
      (parallel_generatedRoot_1235.sem hc) hG (hα k) (hαμ k) hak (hpay k) hg' hg hs
      (hrhs (cellAt o j)) (heB _)
    refine encRel_of_rel (hrel.target_eq ?_) (fun a j ↦ by fin_cases a <;> fin_cases j <;> rfl)
      (fun j ↦ by fin_cases j <;> rfl)
    funext i
    rw [smul_smul, ← map_mul]
    congr 2
    ring
  obtain ⟨key, hkey, -⟩ := hcell o (sel o)
  exact ⟨key, hkey, bucket_of_cells (M := 36000) (W := 2) (K := 2) (p := 1) α (fun _ ↦ μ)
    (generatedRoot.w_1226_0 ctx) ν hcell o (parallel_generatedRoot_1236.sem (F.c46.h1236_5 o))
    (sel o) (hsel o) key hkey⟩

set_option maxHeartbeats 16000000 in
/-- **The second bucket of the step-0 smudge PRF** (gather table `w_1225`, state of member
`⌊i / 2⌋` from the first bucket). -/
theorem online0_smudge_bucket1 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {sm sp : ErrorPoly 8} {μ : Fin 36000 → ErrorPoly 8} {α : Fin 2 → ErrorPoly 8}
    (hα : ∀ k, polyNorm (α k) ≤ 1) (hαμ : ∀ k o, polyNorm (α k * μ o) ≤ 1)
    {G : ExactMatrix modulus_0 8 1 25} (hG : gadgetMatrixRuns backend 4096 25 G) {Ba : Nat}
    (ha : ∀ k : Fin 2, Rel sm (labelKeysG hashModel (generatedRoot.w_7_0 ctx) modulus_0 25 labelPre k)
      (fun j ↦ reducePoly modulus_0 8 (α k * unitPair sm j) • G)
      ![generatedRoot.w_257_0 ctx k, generatedRoot.w_262_0 ctx k] Ba)
    (hpay : ∀ k, generatedRoot.w_276_0 ctx k = fun _ _ ↦ reducePoly modulus_0 8 (α k))
    {g' g : ExactMatrix modulus_0 8 1 25} (hg' : gadgetMatrixRuns backend 4096 25 g')
    (hg : centeredRebaseRuns g' g) {Bs : Nat}
    (hstate : ∀ o : Fin 36000, EncRel sm (generatedRoot.w_1236_0 ctx o, (generatedRoot.w_1236_1 ctx o, (generatedRoot.w_1236_2 ctx o, (generatedRoot.w_1236_3 ctx o, (generatedRoot.w_1236_4 ctx o, (generatedRoot.w_1236_5 ctx o, ()))))))
      (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * μ o * unitPair sp j) • g) Bs)
    {ν : Fin 72000 → ErrorPoly 8} {hT : ExactMatrix modulus_0 8 1 25}
    {eB : Fin 72000 → Fin 2 → ErrorMatrix 8 1 25} {BB : Nat}
    (hrhs : ∀ (i : Fin 72000) j, reducePoly modulus_0 8 sp •
        topPairG hashModel (generatedRoot.w_26_0 ctx) modulus_0 25 rhsPre1b (prfTopTags prfA0 prfB0 i) j +
        ![generatedRoot.w_1237_0 ctx i, generatedRoot.w_1238_0 ctx i] j =
      reducePoly modulus_0 8 (((1 : Nat) : Int) * ν i * unitPair sp j) • hT +
        reduceMatrix modulus_0 8 1 25 (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB)
    (sel : Fin 36000 → Fin 2)
    (hsel : ∀ o j, ν (cellAt o j) = if j = sel o then (1 : ErrorPoly 8) else 0)
    (o : Fin 36000) :
    ∃ key : Fin 2, ((key : Nat) : Int) = generatedRoot.w_1225_0 ctx (cellAt o (sel o)) ∧
      EncRel sm (generatedRoot.w_1240_0 ctx o, (generatedRoot.w_1240_1 ctx o, (generatedRoot.w_1240_2 ctx o, (generatedRoot.w_1240_3 ctx o, (generatedRoot.w_1240_4 ctx o, (generatedRoot.w_1240_5 ctx o, ()))))))
        (fun i ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * (α key * μ o) * unitPair sp i) • hT)
        (2 * ((25 + 25) * 8 * ((25 + 25) * 8 * Ba * 2048 + 8 * 1 * Bs) * 2048 + 8 * 1 * BB)) := by
  have hcell : ∀ (o : Fin 36000) (j : Fin 2), ∃ k : Fin 2,
      ((k : Nat) : Int) = generatedRoot.w_1225_0 ctx (cellAt o j) ∧
      EncRel sm (Tup.ev (k := 6) (generatedRoot.w_1239_0 ctx, (generatedRoot.w_1239_1 ctx, (generatedRoot.w_1239_2 ctx, (generatedRoot.w_1239_3 ctx, (generatedRoot.w_1239_4 ctx, (generatedRoot.w_1239_5 ctx, ())))))) (cellAt o j))
        (fun i ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * (α k * μ o * ν (cellAt o j)) *
          unitPair sp i) • hT)
        ((25 + 25) * 8 * ((25 + 25) * 8 * Ba * 2048 + 8 * 1 * Bs) * 2048 + 8 * 1 * BB) := by
    intro o j
    obtain ⟨x0, x1, xp, -, -, g0, -, -, g1, -, -, g2, hc⟩ := F.c47.h1239_5 (cellAt o j)
    obtain ⟨k, hk, rfl⟩ := g0
    obtain ⟨k1, hk1, rfl⟩ := g1
    obtain ⟨k2, hk2, rfl⟩ := g2
    have e1 : k = k1 := Fin.ext (by omega)
    have e2 : k = k2 := Fin.ext (by omega)
    subst e1 e2
    refine ⟨k, hk, ?_⟩
    have hak := ha k
    rw [hk] at hak
    have hio : (cellAt o j : Nat) / 2 < 36000 := by
      have := o.isLt; have := j.isLt; simp only [cellAt]; omega
    have hoo : (⟨(cellAt o j : Nat) / 2, hio⟩ : Fin 36000) = o := by
      apply Fin.ext; have := j.isLt; simp only [cellAt]; omega
    have hs := hstate o
    rw [← hoo] at hs
    unfold EncRel at hs
    have hrel := PrfCellFlat.rel_1 (by norm_num) hq1 layout hlayout hexact
      (parallel_generatedRoot_1239.sem hc) hio hG (hα k) (by rw [hoo]; exact hαμ k o) hak (hpay k)
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
  exact ⟨key, hkey, bucket_of_cells (M := 36000) (W := 2) (K := 2) (p := 1) α μ
    (generatedRoot.w_1225_0 ctx) ν hcell o (parallel_generatedRoot_1240.sem (F.c47.h1240_5 o))
    (sel o) (hsel o) key hkey⟩

set_option maxHeartbeats 32000000 in
/-- **The step-0 smudge row.** As `online0_fresh_row`: readout LUT `parallel_generatedRoot_1241`,
placement `parallel_generatedRoot_1242`, column `0` from `sequential_generatedRoot_1261`, column
loop `sequential_generatedRoot_1273` (4500 columns). -/
theorem online0_smudge_row {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {sm sp : ErrorPoly 8} (hsm : polyNorm sm ≤ 1)
    {g' g : ExactMatrix modulus_0 8 1 25} (hg' : gadgetMatrixRuns backend 4096 25 g')
    (hg : centeredRebaseRuns g' g) {μ : Fin 36000 → ErrorPoly 8} {B : Nat}
    (hin : ∀ o : Fin 36000, EncRel sm (generatedRoot.w_1240_0 ctx o, (generatedRoot.w_1240_1 ctx o, (generatedRoot.w_1240_2 ctx o, (generatedRoot.w_1240_3 ctx o, (generatedRoot.w_1240_4 ctx o, (generatedRoot.w_1240_5 ctx o, ()))))))
      (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * μ o * unitPair sp j) • g) B)
    {Bh : Nat}
    (hhelp : ∀ i : Fin 8, Rel sm (keyPairG hashModel (generatedRoot.w_7_0 ctx) modulus_0 25 helperPre
        [.integer (sigmaOf sigmas8 i)])
      (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) *
        unitPair (substitution (sigmaOf sigmas8 i).toNat (sigmas8_odd i) sm) j) • g)
      ![generatedRoot.w_13_0 ctx i, generatedRoot.w_14_0 ctx i] Bh)
    {Δ' : ErrorPoly 8} {ν : Fin 36000 → ErrorPoly 8} {gq : ExactMatrix modulus_0 8 1 25}
    {C : Fin 8 → ExactPoly modulus_0 8} {BB : Nat}
    (hswitch : ∀ i : Fin 8, ∃ e : Fin 2 → ErrorMatrix 8 1 25, (∀ j, CoeffBound (e j) BB) ∧ ∀ j,
      reducePoly modulus_0 8 (substitution (sigmaOf sigmas8 i).toNat (sigmas8_odd i) sp) •
          ![![generatedRoot.w_1221_0 ctx, generatedRoot.w_1222_0 ctx],
            ![generatedRoot.w_1223_0 ctx, generatedRoot.w_1224_0 ctx]] 0 j i +
        ![![generatedRoot.w_1221_0 ctx, generatedRoot.w_1222_0 ctx],
          ![generatedRoot.w_1223_0 ctx, generatedRoot.w_1224_0 ctx]] 1 j i =
      (C i * reducePoly modulus_0 8 (Δ' * unitPair sp j)) • gq + reduceMatrix modulus_0 8 1 25 (e j))
    (hμ : ∀ o i, polyNorm (substitution (sigmaOf sigmas8 i).toNat (sigmas8_odd i) (μ o)) ≤ 1)
    (hwalsh : ∀ o, ∑ i, reducePoly modulus_0 8 (substitution (sigmaOf sigmas8 i).toNat
      (sigmas8_odd i) (μ o)) * C i = reducePoly modulus_0 8 (ν o))
    (cc : Fin 4500) :
    EncRel sm (blkT (r := 1 * 1) (C := 4500) (m := 25) (generatedRoot.w_1273_0 ctx, (generatedRoot.w_1273_1 ctx, (generatedRoot.w_1273_2 ctx, (generatedRoot.w_1273_3 ctx, (generatedRoot.w_1273_4 ctx, (generatedRoot.w_1273_5 ctx, ())))))) cc)
      (∑ j : Fin 8, (fun (f : Fin 36000) (jj : Fin 2) ↦ reducePoly modulus_0 8 (Xe (n := 8) (f.val % 8)) •
          (reducePoly modulus_0 8 (Δ' * ν f * unitPair sp jj) • gq))
        ⟨8 * cc + j, by have := cc.isLt; have := j.isLt; omega⟩)
      (8 * (7 * ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048) +
        ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048) + 8 * (8 * 1 * BB))) := by
  -- Every member's readout LUT.
  have hmem : ∀ o : Fin 36000, EncRel sm (generatedRoot.w_1241_0 ctx o, (generatedRoot.w_1241_1 ctx o, (generatedRoot.w_1241_2 ctx o, (generatedRoot.w_1241_3 ctx o, (generatedRoot.w_1241_4 ctx o, (generatedRoot.w_1241_5 ctx o, ()))))))
      (fun j ↦ reducePoly modulus_0 8 (Δ' * ν o * unitPair sp j) • gq)
      (7 * ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048) +
        ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048) + 8 * (8 * 1 * BB)) := by
    intro o
    have he := hin o
    unfold EncRel at he
    have hr := LutMemberFlat.rel (by norm_num) (log := 3) (by norm_num) (L := 7) (by norm_num) hq1
      layout hlayout hexact (fun i a b hb ↦ sequential_parallel_generatedRoot_1241_18.sem hb)
      (parallel_generatedRoot_1241.sem (F.c47.h1241_5 o)) sigmas8_odd hsm hg' hg he
      hhelp hswitch (hμ o) (hwalsh o)
    exact encRel_of_rel hr (fun a j ↦ by fin_cases a <;> fin_cases j <;> rfl)
      (fun j ↦ by fin_cases j <;> rfl)
  -- Placed at `X^(o mod 8)`.
  have hplace : ∀ o : Fin 36000, EncRel sm (generatedRoot.w_1242_0 ctx o, (generatedRoot.w_1242_1 ctx o, (generatedRoot.w_1242_2 ctx o, (generatedRoot.w_1242_3 ctx o, (generatedRoot.w_1242_4 ctx o, (generatedRoot.w_1242_5 ctx o, ()))))))
      (fun j ↦ reducePoly modulus_0 8 (Xe (n := 8) (o.val % 8)) •
        (reducePoly modulus_0 8 (Δ' * ν o * unitPair sp j) • gq))
      (7 * ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048) +
        ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048) + 8 * (8 * 1 * BB)) := fun o ↦
    place_rel (by norm_num) (parallel_generatedRoot_1242.sem (F.c47.h1242_5 o)) (hmem o)
      (by rw [Int.fmod_eq_emod_of_nonneg _ (by norm_num)]; omega) (by omega)
  -- Column `0`: the members `0 … 7`.
  have hx0 := member_prefix (α := ExactMatrix modulus_0 8 1 25) (k := 6) (F := 36000) (Wm := 8) (by norm_num) (by norm_num)
    (fams := (generatedRoot.w_1242_0 ctx, (generatedRoot.w_1242_1 ctx, (generatedRoot.w_1242_2 ctx, (generatedRoot.w_1242_3 ctx, (generatedRoot.w_1242_4 ctx, (generatedRoot.w_1242_5 ctx, ())))))))
    (a := (generatedRoot.w_1243_0 ctx, (generatedRoot.w_1246_0 ctx, (generatedRoot.w_1249_0 ctx, (generatedRoot.w_1252_0 ctx, (generatedRoot.w_1255_0 ctx, (generatedRoot.w_1258_0 ctx, ())))))))
    (b := (generatedRoot.w_1244_0 ctx, (generatedRoot.w_1247_0 ctx, (generatedRoot.w_1250_0 ctx, (generatedRoot.w_1253_0 ctx, (generatedRoot.w_1256_0 ctx, (generatedRoot.w_1259_0 ctx, ())))))))
    (fun c ↦ by
      fin_cases c
      exacts [F.c47.h1243_0, F.c47.h1246_0, F.c47.h1249_0, F.c47.h1252_0, F.c47.h1255_0, F.c47.h1258_0])
    (fun c ↦ by
      fin_cases c
      exacts [F.c47.h1244_0, F.c47.h1247_0, F.c47.h1250_0, F.c47.h1253_0, F.c47.h1256_0, F.c47.h1259_0])
    (fun c ↦ by fin_cases c <;> rfl) F.c47.h1261_5
    (fun i cur next hb ↦ sequential_generatedRoot_1261.sem hb)
  -- The column loop.
  exact column_total_rel (C := 4500) (m := 25) (W := 8) (F := 36000) (by norm_num) (by norm_num) hx0
    (fun c ↦ by
      fin_cases c
      exacts [F.c47.h1262_0, F.c47.h1264_0, F.c47.h1266_0, F.c47.h1268_0, F.c48.h1270_0, F.c48.h1272_0])
    F.c48.h1273_5
    -- Elaborated before unifying with  (otherwise the unifier unfolds the spec).
    (fun i cur next hb ↦ by have hc := sequential_generatedRoot_1273.sem hb; exact hc)
    (t := fun (f : Fin 36000) (jj : Fin 2) ↦ reducePoly modulus_0 8 (Xe (n := 8) (f.val % 8)) •
      (reducePoly modulus_0 8 (Δ' * ν f * unitPair sp jj) • gq))
    (fun o ↦ hplace o) cc

end Dio2Proof
