import ExponentReadoutSetup
import ExponentTrace
import ExponentMonomial

namespace MxxExponentLut
open Mxx.Primitives MxxRuntime
open scoped BigOperators

/-- Native readout plans in `noise_refresh/setup.rs` use `bounded_matrix`
with the typed balanced digit bound, for target, helper and normalization
operands alike. This is the exact gain of that simulator role. -/
theorem readout_decomposition_gain {q n rows inner columns : Nat}
    (hq : 1 < q) (hn : 0 < n) {backend : BackendContext} {base digits : Int}
    {target : ExactMatrix q n rows columns} {decomposition : ExactMatrix q n inner columns}
    (hrun : gadgetDecomposeRuns backend base digits target decomposition) :
    ∃ action : ErrorMatrix n inner columns,
      decomposition = reduceMatrix q n inner columns action ∧
      CoeffBound action (base.toNat / 2) ∧
      ActionGain action (inner * n * (base.toNat / 2)) := by
  obtain ⟨layout, _, hb, _, hw, hd⟩ := hrun
  subst inner
  simp only [castMatrixRows] at hd
  rw [hd, hb]
  obtain ⟨action, ha, hbound⟩ := regularDecomposeMatrix_bounded layout target hq hn
  exact ⟨action, ha, hbound, bounded_matrix_action_gain hbound⟩

private theorem cast_integer_entries {q p n rows columns : Nat} (h : q = p)
    (entries : Matrix (Fin rows) (Fin columns) Int) :
    castExactMatrix h (fun r c => (entries r c : ExactPoly q n)) =
      (fun r c => (entries r c : ExactPoly p n)) := by
  cases h
  rfl

/-- Actual CRT gadget entries are integer constants in the polynomial ring. -/
theorem regular_gadget_integer_entries {q n rows : Nat} (hq : 1 < q) (layout : RegularLayout q) :
    ∃ entries : Matrix (Fin rows)
      (Fin (rows * (layout.crtModuli.length * layout.digitsPerTower))) Int,
      regularGadgetMatrix (n := n) (rows := rows) layout =
        (fun r c => (entries r c : ExactPoly q n)) := by
  classical
  letI : NeZero layout.ordered.modulusProduct := ⟨by rw [layout.ordered.modulusProduct_eq_q]; change q ≠ 0; omega⟩
  let entries := fun (r : Fin rows)
      (c : Fin (rows * (layout.crtModuli.length * layout.digitsPerTower))) =>
    if r = ((regularIndexEquiv layout rows).symm c).1 then
      Int.ofNat (CrtDecomposition.regularGadgetWeight layout.ordered
        ((regularIndexEquiv layout rows).symm c).2).val else 0
  refine ⟨entries, ?_⟩
  unfold regularGadgetMatrix
  have hbefore : (fun r c => regularGadgetUnflattened (n := n) layout r
        ((regularIndexEquiv layout rows).symm c)) =
      (fun r c => (entries r c : ExactPoly layout.ordered.modulusProduct n)) := by
    ext r c
    dsimp only [regularGadgetUnflattened, entries]
    split_ifs
    · simpa only [map_natCast, Int.ofNat_eq_natCast, Int.cast_natCast] using
        (congrArg (algebraMap (ZMod layout.ordered.modulusProduct)
          (ExactPoly layout.ordered.modulusProduct n)) (ZMod.natCast_zmod_val
            (CrtDecomposition.regularGadgetWeight layout.ordered
              ((regularIndexEquiv layout rows).symm c).2))).symm
    · simp
  rw [hbefore]
  exact cast_integer_entries _ entries

/-- Actual gadget construction is fixed by every valid odd automorphism. -/
theorem readout_gadget_automorphism {q n rows columns : Nat}
    (hq : 1 < q) (hn : 0 < n) {backend : BackendContext} {base digits : Int}
    {gadget : ExactMatrix q n rows columns} (index : Nat) (hodd : Odd index)
    (hrun : gadgetMatrixRuns backend base digits gadget) :
    ringAutomorphism index gadget = gadget := by
  obtain ⟨layout, _, _, _, hw, hg⟩ := hrun
  subst columns
  simp only [castMatrixColumns] at hg
  obtain ⟨entries, he⟩ := regular_gadget_integer_entries (n := n) (rows := rows) hq layout
  rw [hg, he]
  exact ringAutomorphism_integer_constants hq hn index hodd entries

/-- Coefficient L1 is subadditive, retaining the simulator's sharp scalar gain. -/
theorem polynomialL1_add_le {n : Nat} (a b : ErrorPoly n) :
    polynomialL1 (a + b) ≤ polynomialL1 a + polynomialL1 b := by
  unfold polynomialL1
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_le_sum
  intro i _
  simpa only [Negacyclic.coeff_add] using Int.natAbs_add_le (a.coeff i) (b.coeff i)

theorem polynomialL1_sum_le {n : Nat} {α : Type} [DecidableEq α]
    (members : Finset α) (f : α → ErrorPoly n) :
    polynomialL1 (∑ i ∈ members, f i) ≤ ∑ i ∈ members, polynomialL1 (f i) := by
  induction members using Finset.induction_on with
  | empty => simp [polynomialL1]
  | @insert i members hi ih =>
    simp only [Finset.sum_insert hi]
    exact (polynomialL1_add_le _ _).trans (Nat.add_le_add_left ih _)

private theorem polynomialL1_scaled_root {n : Nat} (hn : 0 < n) (a : Int)
    (exponent : Nat) :
    polynomialL1 (algebraMap Int (ErrorPoly n) a *
      AdjoinRoot.root (negacyclicModulus n Int) ^ exponent) = a.natAbs := by
  have hr : AdjoinRoot.root (negacyclicModulus n Int) ^ exponent =
      algebraMap Int (ErrorPoly n) ((-1 : Int) ^ (exponent / n)) *
        AdjoinRoot.root (negacyclicModulus n Int) ^ (exponent % n) := by
    conv_lhs => rw [← Nat.div_add_mod exponent n]
    rw [pow_add, pow_mul, Negacyclic.root_pow_n]
    simp
  rw [hr, ← mul_assoc, ← map_mul]
  change polynomialL1 (scaledBasis (a * (-1) ^ (exponent / n))
    (⟨exponent % n, Nat.mod_lt _ hn⟩ : Fin n)) = _
  rw [polynomialL1_scaledBasis hn]
  simp [Int.natAbs_mul, Int.natAbs_pow]

theorem polynomialL1_mul_le {n : Nat} (hn : 0 < n) (a b : ErrorPoly n) :
    polynomialL1 (a * b) ≤ polynomialL1 a * polynomialL1 b := by
  have expansion : a * b = ∑ i : Fin n, ∑ j : Fin n,
      algebraMap Int (ErrorPoly n) (a.coeff i * b.coeff j) *
        AdjoinRoot.root (negacyclicModulus n Int) ^ (i.val + j.val) := by
    conv_lhs => rw [Negacyclic.expansion hn a, Negacyclic.expansion hn b]
    rw [Finset.sum_mul_sum]
    apply Finset.sum_congr rfl
    intro i _
    apply Finset.sum_congr rfl
    intro j _
    rw [map_mul, pow_add]
    ring
  rw [expansion]
  calc
    _ ≤ ∑ i : Fin n, polynomialL1 (∑ j : Fin n,
        algebraMap Int (ErrorPoly n) (a.coeff i * b.coeff j) *
          AdjoinRoot.root (negacyclicModulus n Int) ^ (i.val + j.val)) :=
      polynomialL1_sum_le _ _
    _ ≤ ∑ i : Fin n, ∑ j : Fin n, (a.coeff i).natAbs * (b.coeff j).natAbs := by
      apply Finset.sum_le_sum
      intro i _
      simpa only [polynomialL1_scaled_root hn, Int.natAbs_mul] using
        polynomialL1_sum_le Finset.univ (fun j : Fin n =>
          algebraMap Int (ErrorPoly n) (a.coeff i * b.coeff j) *
            AdjoinRoot.root (negacyclicModulus n Int) ^ (i.val + j.val))
    _ = _ := by simp only [polynomialL1, Finset.sum_mul_sum]

theorem errorAutomorphism_polynomialL1_le {n : Nat} (hn : 0 < n)
    (index : Nat) (input : ErrorPoly n) :
    polynomialL1 (errorAutomorphism index input) ≤ polynomialL1 input := by
  unfold errorAutomorphism
  simpa only [polynomialL1_scaled_root hn] using
    polynomialL1_sum_le Finset.univ (fun i : Fin n =>
      algebraMap Int (ErrorPoly n) (input.coeff i) *
        AdjoinRoot.root (negacyclicModulus n Int) ^ (i.val * index))

/-- Signed monomial multiplication and actual substitution preserve the target
polynomial's L1 bound; no Boolean-table assumption is required. -/
theorem readout_payload_lift {q n gain : Nat} (hq : 1 < q) (hn : 0 < n)
    (index : Nat) (hodd : Odd index) {message target : ExactPoly q n}
    (hmessage : SignedMonomial message) (targetLift : ErrorPoly n)
    (htarget : reducePoly q n targetLift = target) (hgain : polynomialL1 targetLift ≤ gain) :
    ∃ lift : ErrorPoly n,
      readoutAutomorphismHom index hodd (message * target) = reducePoly q n lift ∧
      polynomialL1 lift ≤ gain := by
  obtain ⟨messageLift, hm, hml⟩ := signedMonomial_integer_lift hn hmessage
  refine ⟨errorAutomorphism index (messageLift * targetLift), ?_, ?_⟩
  · have h := congrFun (congrFun
      (errorAutomorphism_reduce (rows := 1) (columns := 1) hq hn index
        (fun _ _ => messageLift * targetLift)) 0) 0
    rw [readoutAutomorphismHom_runtime hq hn index hodd] at h
    simpa only [reduceMatrix, map_mul, hm, htarget] using h.symm
  · exact (errorAutomorphism_polynomialL1_le hn _ _).trans
      ((polynomialL1_mul_le hn _ _).trans
        ((Nat.mul_le_mul hml hgain).trans_eq (one_mul gain)))

/-- Native readout's exact encoding residual, derived from sampled switch and
mask equations and both cache reconstructions. Only the source encoding is an
approximation premise. The scalar target gain is its actual integer L1. -/
theorem readout_encoding_transfer
    {q log rows key columns inputBound cutoff targetGain normalizationGain plainGain : Nat}
    {α : Type} [DecidableEq α] (hq : 1 < q) (members : Finset α)
    (source : ExactMatrix q (2 ^ log) rows columns)
    (secret setup : ExactMatrix q (2 ^ log) rows key)
    (publicKey gadget : ExactMatrix q (2 ^ log) key columns)
    (message targetScalar inverseScalar : ExactPoly q (2 ^ log))
    (targetLift : ErrorPoly (2 ^ log))
    (target normalization : ErrorMatrix (2 ^ log) columns columns)
    (sigma : α → Nat) (odd : ∀ i, Odd (sigma i))
    (raw mask : α → ExactMatrix q (2 ^ log) rows columns)
    (maskKey : α → ExactMatrix q (2 ^ log) key columns)
    (action helper : α → ErrorMatrix (2 ^ log) columns columns)
    (switchError maskError : α → ErrorMatrix (2 ^ log) rows columns)
    (actionGain helperGain : α → Nat)
    (hsource : Approx source (secret * publicKey - message • (setup * gadget)) inputBound)
    (hmonomial : SignedMonomial message)
    (htargetLift : reducePoly q (2 ^ log) targetLift = targetScalar)
    (hplain : polynomialL1 targetLift ≤ plainGain)
    (htarget : ActionGain target targetGain)
    (hnormalization : ActionGain normalization normalizationGain)
    (htargetCache : gadget * reduceMatrix q (2 ^ log) columns columns target =
      targetScalar • gadget)
    (hnormalizationCache : gadget * reduceMatrix q (2 ^ log) columns columns normalization =
      inverseScalar • gadget)
    (hgadget : ∀ i ∈ members, ringAutomorphism (sigma i) gadget = gadget)
    (hrun : ∀ i ∈ members, ringAutomorphismRuns (sigma i)
      (source * reduceMatrix q (2 ^ log) columns columns target) (raw i))
    (hmask : ∀ i ∈ members, mask i =
      (secret * maskKey i - ringAutomorphism (sigma i) secret * gadget) +
        reduceMatrix q (2 ^ log) rows columns (maskError i))
    (hswitchCache : ∀ i ∈ members,
      ringAutomorphism (sigma i) setup * gadget *
          reduceMatrix q (2 ^ log) columns columns (action i) =
        setup * gadget + reduceMatrix q (2 ^ log) rows columns (switchError i))
    (halign : ∀ i ∈ members, gadget * reduceMatrix q (2 ^ log) columns columns (helper i) =
      ringAutomorphism (sigma i)
        (publicKey * reduceMatrix q (2 ^ log) columns columns target) *
          reduceMatrix q (2 ^ log) columns columns (action i))
    (hswitch : ∀ i ∈ members, CoeffBound (switchError i) cutoff)
    (hmaskError : ∀ i ∈ members, CoeffBound (maskError i) cutoff)
    (haction : ∀ i ∈ members, ActionGain (action i) (actionGain i))
    (hhelper : ∀ i ∈ members, ActionGain (helper i) (helperGain i)) :
    Approx
      ((source * reduceMatrix q (2 ^ log) columns columns target +
          ∑ i ∈ members, (mask i * reduceMatrix q (2 ^ log) columns columns (helper i) +
            raw i * reduceMatrix q (2 ^ log) columns columns (action i))) *
        reduceMatrix q (2 ^ log) columns columns normalization)
      (secret * ((publicKey * reduceMatrix q (2 ^ log) columns columns target +
          ∑ i ∈ members, maskKey i * reduceMatrix q (2 ^ log) columns columns (helper i)) *
            reduceMatrix q (2 ^ log) columns columns normalization) -
        (inverseScalar * (message * targetScalar +
          ∑ i ∈ members, readoutAutomorphismHom (sigma i) (odd i) (message * targetScalar))) •
            (setup * gadget))
      (normalizedTraceBound members inputBound cutoff targetGain 1 normalizationGain
        plainGain actionGain helperGain) := by
  have hn : 0 < 2 ^ log := by positivity
  have htargetIdeal :
      (secret * publicKey - message • (setup * gadget)) *
        reduceMatrix q (2 ^ log) columns columns target =
      secret * (publicKey * reduceMatrix q (2 ^ log) columns columns target) -
        (message * targetScalar) • (setup * gadget) := by
    rw [Matrix.sub_mul, Matrix.smul_mul, Matrix.mul_assoc setup, htargetCache,
      Matrix.mul_smul, smul_smul, Matrix.mul_assoc]
  have ht := approximate_right_action hn hsource htarget
  rw [htargetIdeal] at ht
  have hb (i : α) (hi : i ∈ members) :
      Approx (mask i * reduceMatrix q (2 ^ log) columns columns (helper i) +
        raw i * reduceMatrix q (2 ^ log) columns columns (action i))
        (secret * (maskKey i * reduceMatrix q (2 ^ log) columns columns (helper i)) -
          readoutAutomorphismHom (sigma i) (odd i) (message * targetScalar) • (setup * gadget))
        (actionGain i * (targetGain * inputBound) + (plainGain + helperGain i) * cutoff) := by
    obtain ⟨lift, hl, hlg⟩ := readout_payload_lift hq hn (sigma i) (odd i)
      hmonomial targetLift htargetLift hplain
    have hs := readout_setup_of_cache_equations hq hn (sigma i) (odd i) secret setup
      (publicKey * reduceMatrix q (2 ^ log) columns columns target) (maskKey i) gadget
      (action i) (helper i) (switchError i) (message * targetScalar) lift
      (hgadget i hi) (hswitchCache i hi) (halign i hi) hl
    apply readout_branch_transfer (message := -lift) hq ht (hrun i hi) (hmask i hi) _
      (hswitch i hi) (hmaskError i hi) (by simpa using hlg) (haction i hi) (hhelper i hi)
    simpa only [Int.toNat_natCast] using hs
  have hs := approximate_sum members _ _ _ hb
  obtain ⟨et, het, hetb⟩ := ht
  obtain ⟨es, hes, hesb⟩ := hs
  have hsum : Approx
      (source * reduceMatrix q (2 ^ log) columns columns target +
        ∑ i ∈ members, (mask i * reduceMatrix q (2 ^ log) columns columns (helper i) +
          raw i * reduceMatrix q (2 ^ log) columns columns (action i)))
      ((secret * (publicKey * reduceMatrix q (2 ^ log) columns columns target) -
          (message * targetScalar) • (setup * gadget)) +
        ∑ i ∈ members, (secret * (maskKey i * reduceMatrix q (2 ^ log) columns columns (helper i)) -
          readoutAutomorphismHom (sigma i) (odd i) (message * targetScalar) • (setup * gadget)))
      (targetGain * inputBound + ∑ i ∈ members,
        (actionGain i * (targetGain * inputBound) + (plainGain + helperGain i) * cutoff)) := by
    refine ⟨et + es, ?_, coeffBound_add hetb hesb⟩
    rw [het, hes, reduceMatrix_add]
    abel
  have result := approximate_right_action hn hsum hnormalization
  have hideal :
      ((secret * (publicKey * reduceMatrix q (2 ^ log) columns columns target) -
          (message * targetScalar) • (setup * gadget)) +
        ∑ i ∈ members, (secret * (maskKey i * reduceMatrix q (2 ^ log) columns columns (helper i)) -
          readoutAutomorphismHom (sigma i) (odd i) (message * targetScalar) • (setup * gadget))) *
            reduceMatrix q (2 ^ log) columns columns normalization =
      secret * ((publicKey * reduceMatrix q (2 ^ log) columns columns target +
          ∑ i ∈ members, maskKey i * reduceMatrix q (2 ^ log) columns columns (helper i)) *
            reduceMatrix q (2 ^ log) columns columns normalization) -
        (inverseScalar * (message * targetScalar +
          ∑ i ∈ members, readoutAutomorphismHom (sigma i) (odd i) (message * targetScalar))) •
            (setup * gadget) := by
    rw [Finset.sum_sub_distrib, ← Matrix.mul_sum, ← Finset.sum_smul,
      sub_add_sub_comm, ← Matrix.mul_add, ← add_smul]
    rw [Matrix.sub_mul, Matrix.mul_assoc, Matrix.smul_mul,
      Matrix.mul_assoc setup, hnormalizationCache, Matrix.mul_smul, smul_smul,
      mul_comm _ inverseScalar]
  rw [hideal] at result
  simpa only [normalizedTraceBound, one_mul] using result

end MxxExponentLut
