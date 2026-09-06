import DiamondProofParameters
import DiamondInjectorWitness
import DiamondIndexProof
import DiamondSelectorProof
import DiamondAccumulatedSecretProof
import DiamondSelectorWitness

open Mxx.Primitives MxxRuntime
open DiamondProofParameters

namespace DiamondGeneratedProof

theorem producer_transition_bounded
    (backend : BackendContext) (hashModel : HashModel) (params : Stage_encrypt.Params)
    (message : Bool) (initial : ExactMatrix q n 1 inner)
    (transitions : Fin transitionCount → ExactMatrix q n inner inner)
    (producer : InjectorRootWitness backend hashModel params message initial transitions)
    (slot : Fin transitionCount) :
    PreimageWithin (transitions slot) params.diamond_preimage_max_coefficient_bound.toNat := by
  obtain ⟨position, _, _, hsourceTrapdoor⟩ := generated_source_pool_lookup backend hashModel
    params slot.val _ _ _ _ _ (producer.sourcesRun slot)
  have hsample := producer.basesRun position
  rcases hsample with ⟨trapdoor, matrixValue, hsample, hout⟩
  have hpool : producer.trapdoors position = trapdoor :=
    congrArg (fun value ↦ value.2.1) hout
  have hkind : (producer.sourceTrapdoors slot).kind = .sampledSecret := by
    rw [hsourceTrapdoor, hpool]
    exact trapdoorSample_sampled hsample
  rcases producer.preimagesRun slot with ⟨value, _, hdispatch, hvalue⟩
  rcases hdispatch.2 with hsampled | hpublic
  · exact hvalue.symm ▸ preimageRuns_bounded hsampled
  · have hbad := hkind.symm.trans hpublic.1
    cases hbad

/-- The four possible second-column shapes of the actual injector selectors. -/
def SparseSelectorColumn (secret : ExactPoly q n)
    (selector : ExactMatrix q n 2 2) : Prop :=
  (selector 0 1 = 0 ∧ selector 1 1 = secret) ∨
  (selector 0 1 = 0 ∧ selector 1 1 = 1) ∨
  (selector 0 1 = 0 ∧ selector 1 1 = 0) ∨
  (selector 0 1 = secret ∧ selector 1 1 = 0)

theorem generated_selector_scan_sparse
    (backend : BackendContext) (hashModel : HashModel) (params : Stage_encrypt.Params)
    (slot count : Nat) (digit state firstNew : Int)
    (initial output : ExactMatrix q n 2 2) (secret : ExactMatrix q n 1 1)
    (hinitial : SparseSelectorColumn (secret 0 0) initial)
    (hrun : MxxIR.IterRuns
      (fun bit current next ↦ Stage_encrypt.sequential_parallel_generatedRoot_72_21
        backend hashModel params slot bit (current, digit, state, firstNew, secret, ()) next)
      count initial output) : SparseSelectorColumn (secret 0 0) output := by
  apply MxxIR.IterRuns.invariant
    (Invariant := fun _ value ↦ SparseSelectorColumn (secret 0 0) value) hinitial _ hrun
  intro bit current next ih hstep
  by_cases hstate : state = firstNew + Int.ofNat bit
  · obtain ⟨bitValue, hbitValue, _, h01, _, h11⟩ := generated_selector_match backend
      hashModel params slot bit digit state firstNew current next secret hstate hstep
    rcases hbitValue with ⟨position, _, hvalue⟩
    fin_cases position
    · have hb : bitValue = 0 := hvalue
      exact Or.inr (Or.inr (Or.inl ⟨by simpa [hb] using h01, h11⟩))
    · have hb : bitValue = 1 := hvalue
      exact Or.inr (Or.inr (Or.inr ⟨by simpa [hb] using h01, h11⟩))
  · have heq := generated_selector_no_match backend hashModel params slot bit digit state
      firstNew current next secret hstate hstep
    exact heq.symm ▸ ih

theorem selector_witness_integer_error
    (backend : BackendContext) (hashModel : HashModel) (params : Stage_encrypt.Params)
    (slot : Nat) (secret : ExactMatrix q n 1 1)
    (publicMatrix target : ExactMatrix q n 2 inner)
    (witness : InjectorSelectorWitness backend hashModel params slot secret publicMatrix target) :
    ∃ (selector : ExactMatrix q n 2 2) (error : ErrorMatrix n 2 inner),
      selector = witness.selector ∧
      selector 0 0 = secret 0 0 ∧ selector 1 0 = 0 ∧
      SparseSelectorColumn (secret 0 0) selector ∧
      CoeffBound error params.diamond_error_max_coefficient_bound.toNat ∧
      target = selector * publicMatrix + reduceMatrix q n 2 inner error := by
  rcases witness with ⟨regular, initialZero, initial, selector, error, state, digit, firstNew,
    _, _, _, hregular, hzero, hselect, hscan, herror, htarget⟩
  have hr : regular 0 0 = secret 0 0 ∧ regular 1 0 = 0 ∧
      SparseSelectorColumn (secret 0 0) regular := by
    refine ⟨?_, ?_, Or.inl ⟨?_, ?_⟩⟩
    all_goals simpa [concatDiagonal] using hregular _ _
  have hz : initialZero 0 0 = secret 0 0 ∧ initialZero 1 0 = 0 ∧
      SparseSelectorColumn (secret 0 0) initialZero := by
    refine ⟨?_, ?_, Or.inr (Or.inl ⟨?_, ?_⟩)⟩
    all_goals simpa [concatDiagonal] using hzero _ _
  have hi : initial 0 0 = secret 0 0 ∧ initial 1 0 = 0 ∧
      SparseSelectorColumn (secret 0 0) initial := by
    rcases hselect with ⟨position, _, hvalue⟩
    fin_cases position
    · exact hvalue.symm ▸ hr
    · exact hvalue.symm ▸ hz
  obtain ⟨h00, h10⟩ := generated_selector_scan_first_column backend hashModel params slot
    _ _ _ _ initial selector secret ⟨hi.1, hi.2.1⟩ hscan
  have hsparse := generated_selector_scan_sparse backend hashModel params slot
    _ _ _ _ initial selector secret hi.2.2 hscan
  obtain ⟨errorLift, hreduce, hbound⟩ := herror.2.2.1
  exact ⟨selector, errorLift, rfl, h00, h10, hsparse, hbound, hreduce ▸ htarget⟩

theorem sparse_selector_integer_row
    (row : ErrorMatrix n 1 2) (secret : ErrorPoly n)
    (selector : ExactMatrix q n 2 2) (bound : Nat)
    (hrow : CoeffBound row bound) (hsecret : polyNorm secret ≤ 1)
    (h00 : selector 0 0 = reducePoly q n secret) (h10 : selector 1 0 = 0)
    (hsparse : SparseSelectorColumn (reducePoly q n secret) selector) :
    ∃ next : ErrorMatrix n 1 2,
      next 0 0 = row 0 0 * secret ∧
      reduceMatrix q n 1 2 next = reduceMatrix q n 1 2 row * selector ∧
      CoeffBound next (n * bound) := by
  have hnorm (column : Fin 2) : polyNorm (row 0 column) ≤ bound := by
    apply Finset.sup_le
    intro coefficient _
    exact hrow 0 column coefficient
  have hmul (column : Fin 2) : polyNorm (row 0 column * secret) ≤ n * bound := by
    have h := polyNorm_mul_le_tight (by decide : 0 < n) (row 0 column) secret
    have hm := Nat.mul_le_mul (Nat.mul_le_mul_left n (hnorm column)) hsecret
    exact h.trans (by simpa using hm)
  have hbuild (second : ErrorPoly n) (hsecond : polyNorm second ≤ n * bound)
      (heq : reducePoly q n second =
        reducePoly q n (row 0 0) * selector 0 1 +
        reducePoly q n (row 0 1) * selector 1 1) :
      ∃ next : ErrorMatrix n 1 2,
        next 0 0 = row 0 0 * secret ∧
        reduceMatrix q n 1 2 next = reduceMatrix q n 1 2 row * selector ∧
        CoeffBound next (n * bound) := by
    refine ⟨fun _ column ↦ if column = 0 then row 0 0 * secret else second, rfl, ?_, ?_⟩
    · funext i j
      fin_cases i
      fin_cases j
      · simp [reduceMatrix, Matrix.mul_apply, Fin.sum_univ_two, h00, h10]
      · simpa [reduceMatrix, Matrix.mul_apply, Fin.sum_univ_two] using heq
    · intro i j coefficient
      fin_cases i
      fin_cases j
      · exact (coeff_natAbs_le_polyNorm _ _).trans (hmul 0)
      · exact (coeff_natAbs_le_polyNorm _ _).trans hsecond
  rcases hsparse with ⟨h01, h11⟩ | ⟨h01, h11⟩ | ⟨h01, h11⟩ | ⟨h01, h11⟩
  · exact hbuild (row 0 1 * secret) (hmul 1) (by simp [h01, h11])
  · exact hbuild (row 0 1)
      ((hnorm 1).trans (Nat.le_mul_of_pos_left bound (by decide : 0 < n)))
      (by simp [h01, h11])
  · exact hbuild 0 (by simp) (by simp [h01, h11])
  · exact hbuild (row 0 0 * secret) (hmul 0) (by simp [h01, h11])

theorem producer_integer_digit_samples
    (backend : BackendContext) (hashModel : HashModel) (params : Stage_encrypt.Params)
    (message : Bool) (initial : ExactMatrix q n 1 inner)
    (transitions : Fin transitionCount → ExactMatrix q n inner inner)
    (producer : InjectorRootWitness backend hashModel params message initial transitions) :
    ∃ samples : Fin sampleCount → ErrorPoly n, ∀ position,
      producer.digitSamples position 0 0 = reducePoly q n (samples position) ∧
      polyNorm (samples position) ≤ 1 := by
  have hex (position : Fin sampleCount) : ∃ sample : ErrorPoly n,
      producer.digitSamples position 0 0 = reducePoly q n sample ∧
      polyNorm sample ≤ 1 := by
    rcases producer.samplesRun position with ⟨value, hsample, hvalue⟩
    obtain ⟨lift, hreduce, hbound⟩ := hsample.2
    refine ⟨lift 0 0, ?_, ?_⟩
    · rw [hvalue, hreduce]
      rfl
    · apply Finset.sup_le
      intro coefficient _
      have h := hbound 0 0 coefficient
      omega
  choose samples hsamples using hex
  exact ⟨samples, hsamples⟩

theorem producer_integer_selected_secret
    (backend : BackendContext) (hashModel : HashModel) (params : Stage_encrypt.Params)
    (message : Bool) (initial : ExactMatrix q n 1 inner)
    (transitions : Fin transitionCount → ExactMatrix q n inner inner)
    (producer : InjectorRootWitness backend hashModel params message initial transitions)
    (samples : Fin sampleCount → ErrorPoly n)
    (hsamples : ∀ position,
      producer.digitSamples position 0 0 = reducePoly q n (samples position) ∧
      polyNorm (samples position) ≤ 1)
    (slot : Fin transitionCount) :
    ∃ position : Fin sampleCount, (position.val : Int) = producer.digitIndices slot ∧
      producer.digitSecrets slot 0 0 = reducePoly q n (samples position) ∧
      polyNorm (samples position) ≤ 1 := by
  rcases producer.secretsRun slot with ⟨value, _, _, ⟨position, hindex, hvalue⟩, hout⟩
  refine ⟨position, hindex, ?_, (hsamples position).2⟩
  rw [hout, hvalue]
  exact (hsamples position).1

theorem generated_integer_transition
    (backend : BackendContext) (hashModel : HashModel)
    (params : Stage_encrypt.Params) (decryptParams : Stage_decrypt.Params)
    (message : Bool) (initial : ExactMatrix q n 1 inner)
    (transitions : Fin transitionCount → ExactMatrix q n inner inner)
    (producer : InjectorRootWitness backend hashModel params message initial transitions)
    (slot : Fin transitionCount) (layer lane : Nat)
    (row : ErrorMatrix n 1 2) (error : ErrorMatrix n 1 inner)
    (secret : ErrorPoly n) (rowBound errorBound : Nat)
    (current next : ExactMatrix q n 1 inner)
    (hsecret : producer.digitSecrets slot 0 0 = reducePoly q n secret)
    (hsecretBound : polyNorm secret ≤ 1)
    (hrow : CoeffBound row rowBound) (herror : CoeffBound error errorBound)
    (hcurrent : current = reduceMatrix q n 1 2 row * producer.sourcePublics slot +
      reduceMatrix q n 1 inner error)
    (hstep : Stage_decrypt.parallel_sequential_generatedRoot_8_13 backend decryptParams
      layer lane (current, transitions slot, ()) next) :
    ∃ (selectorWitness : InjectorSelectorWitness backend hashModel params slot.val
        (producer.digitSecrets slot) (producer.targetPublics slot) (producer.targets slot))
      (nextRow : ErrorMatrix n 1 2) (nextError : ErrorMatrix n 1 inner),
      reduceMatrix q n 1 2 nextRow =
        reduceMatrix q n 1 2 row * selectorWitness.selector ∧
      nextRow 0 0 = row 0 0 * secret ∧
      next = reduceMatrix q n 1 2 nextRow * producer.targetPublics slot +
        reduceMatrix q n 1 inner nextError ∧
      CoeffBound nextRow (n * rowBound) ∧
      CoeffBound nextError (2 * n * rowBound *
        params.diamond_error_max_coefficient_bound.toNat +
        inner * n * errorBound * params.diamond_preimage_max_coefficient_bound.toNat) := by
  obtain ⟨selectorWitness⟩ := generated_selector_witness backend hashModel params slot.val
    _ _ _ (producer.targetsRun slot)
  obtain ⟨selector, targetError, hselector, h00, h10, hsparse, htargetBound, htarget⟩ :=
    selector_witness_integer_error backend hashModel params slot.val _ _ _ selectorWitness
  rw [hsecret] at h00 hsparse
  obtain ⟨nextRow, hfirst, hrowReduce, hnextRowBound⟩ :=
    sparse_selector_integer_row row secret selector rowBound hrow hsecretBound h00 h10 hsparse
  obtain ⟨preimageLift, hpreimageLift, hpreimageBound⟩ :=
    producer_transition_bounded backend hashModel params message initial transitions producer slot
  have hrelation : producer.sourcePublics slot * transitions slot = producer.targets slot := by
    rcases producer.preimagesRun slot with ⟨value, _, hdispatch, hvalue⟩
    rw [hvalue]
    exact preimageRunsDispatched_equation (by decide) (by decide) hdispatch
  obtain ⟨nextError, hnext, hnextBound⟩ := consume_right_preimage_bound
    (q := q) (inner := inner) (targetColumns := inner) (value := current)
    (by decide : 0 < n) (hcurrent.trans (add_comm _ _)) htarget rfl hpreimageLift hrelation
    hrow htargetBound herror hpreimageBound
  refine ⟨selectorWitness, nextRow, nextError, by simpa only [hselector] using hrowReduce,
    hfirst, ?_, hnextRowBound, hnextBound⟩
  have hstep' : next = current * transitions slot := hstep
  rw [hstep', hnext, hrowReduce, Matrix.mul_assoc]

#print axioms generated_integer_transition
#print axioms producer_integer_digit_samples
#print axioms producer_integer_selected_secret
#print axioms sparse_selector_integer_row
#print axioms selector_witness_integer_error
#print axioms producer_transition_bounded
#print axioms generated_selector_scan_sparse

end DiamondGeneratedProof
