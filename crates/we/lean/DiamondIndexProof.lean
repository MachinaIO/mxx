import DiamondProofParameters
import Stage_encrypt
import Stage_decrypt

open Mxx.Primitives MxxRuntime
open DiamondProofParameters

namespace DiamondGeneratedProof

theorem transition_coordinates (stateCount digitBase layer : Nat)
    (hstates : 0 < stateCount) (hdigits : 0 < digitBase)
    (digit : Fin digitBase) (state : Fin stateCount) :
    ((layer * digitBase + digit.val) * stateCount + state.val) /
        (digitBase * stateCount) = layer ∧
      ((layer * digitBase + digit.val) * stateCount + state.val) % stateCount = state.val ∧
      ((layer * digitBase + digit.val) * stateCount + state.val) / stateCount =
        layer * digitBase + digit.val := by
  have hremainder : digit.val * stateCount + state.val < digitBase * stateCount := by
    have hd := digit.isLt
    have hs := state.isLt
    nlinarith
  have hflat : (layer * digitBase + digit.val) * stateCount + state.val =
      (digit.val * stateCount + state.val) + layer * (digitBase * stateCount) := by ring
  refine ⟨?_, ?_, ?_⟩
  · rw [hflat, Nat.add_mul_div_right _ _ (Nat.mul_pos hdigits hstates),
      Nat.div_eq_of_lt hremainder, Nat.zero_add]
  · rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt state.isLt]
  · rw [Nat.add_comm, Nat.add_mul_div_right _ _ hstates,
      Nat.div_eq_of_lt state.isLt, Nat.zero_add]

theorem generated_target_index
    (backend : BackendContext) (hashModel : HashModel) (params : Stage_encrypt.Params)
    (stateCount digitBase layer : Nat) (hstates : 0 < stateCount) (hdigits : 0 < digitBase)
    (hstateGeometry : (stateCount : Int) =
      1 + params.diamond_batch_bits * params.diamond_input_count)
    (hbaseGeometry : (digitBase : Int) = params.diamond_digit_base)
    (digit : Fin digitBase) (state : Fin stateCount) (index : Int)
    (hrun : Stage_encrypt.parallel_generatedRoot_70 backend hashModel params
      ((layer * digitBase + digit.val) * stateCount + state.val) () index) :
    index = ((layer + 1) * stateCount + state.val : Nat) := by
  have hwidth : ((digitBase * stateCount : Nat) : Int) =
      params.diamond_batch_bits * params.diamond_digit_base * params.diamond_input_count +
        params.diamond_digit_base := by
    rw [Nat.cast_mul, hstateGeometry, hbaseGeometry]
    ring
  have heq := hrun.2.2
  rw [← hwidth, ← hstateGeometry] at heq
  obtain ⟨hlayer, hstate, _⟩ := transition_coordinates stateCount digitBase layer
    hstates hdigits digit state
  simp only [Int.ofNat_eq_natCast, ← Int.natCast_ediv, ← Int.natCast_emod,
    hlayer, hstate] at heq
  simpa only [Nat.cast_add, Nat.cast_mul, Nat.cast_one] using heq

theorem generated_runtime_transition_index
    (backend : BackendContext) (params : Stage_decrypt.Params)
    (stateCount digitBase layer digit state : Nat)
    (hstateGeometry : (stateCount : Int) =
      1 + params.diamond_batch_bits * params.diamond_input_count)
    (hbaseGeometry : (digitBase : Int) = params.diamond_digit_base)
    (index : Int)
    (hrun : Stage_decrypt.parallel_sequential_generatedRoot_8_10 backend params layer state
      (Int.ofNat layer, Int.ofNat digit, ()) index) :
    index = ((layer * digitBase + digit) * stateCount + state : Nat) := by
  have hwidth : ((digitBase * stateCount : Nat) : Int) =
      params.diamond_batch_bits * params.diamond_digit_base * params.diamond_input_count +
        params.diamond_digit_base := by
    rw [Nat.cast_mul, hstateGeometry, hbaseGeometry]
    ring
  dsimp only [Stage_decrypt.parallel_sequential_generatedRoot_8_10] at hrun
  rw [← hwidth, ← hstateGeometry] at hrun
  convert hrun using 1
  simp only [Int.ofNat_eq_natCast, Nat.cast_add, Nat.cast_mul]
  ring

#print axioms generated_runtime_transition_index

theorem generated_source_index_agrees
    (backend : BackendContext) (hashModel : HashModel)
    (params : Stage_encrypt.Params) (decryptParams : Stage_decrypt.Params)
    (stateCount digitBase layer : Nat) (hstates : 0 < stateCount) (hdigits : 0 < digitBase)
    (hstateGeometry : (stateCount : Int) =
      1 + params.diamond_batch_bits * params.diamond_input_count)
    (hbaseGeometry : (digitBase : Int) = params.diamond_digit_base)
    (digit : Fin digitBase) (state : Fin stateCount) (setupIndex sourceIndex : Int)
    (hsetup : Stage_encrypt.parallel_generatedRoot_65 backend hashModel params
      ((layer * digitBase + digit.val) * stateCount + state.val) () setupIndex)
    (hruntime : Stage_decrypt.parallel_sequential_generatedRoot_8_5 backend decryptParams
      layer state.val (Int.ofNat layer * params.diamond_batch_bits + 1) sourceIndex) :
    setupIndex = (layer * stateCount : Nat) + sourceIndex := by
  have hwidth : ((digitBase * stateCount : Nat) : Int) =
      params.diamond_batch_bits * params.diamond_digit_base * params.diamond_input_count +
        params.diamond_digit_base := by
    rw [Nat.cast_mul, hstateGeometry, hbaseGeometry]
    ring
  obtain ⟨hlayer, hstate, _⟩ := transition_coordinates stateCount digitBase layer
    hstates hdigits digit state
  dsimp only [Stage_encrypt.parallel_generatedRoot_65] at hsetup
  rcases hsetup with ⟨setupSelected, _, _, _, _, _, hsetupSelect, hsetupOut⟩
  rw [← hwidth, ← hstateGeometry] at hsetupSelect hsetupOut
  simp only [Int.ofNat_eq_natCast, ← Int.natCast_ediv, ← Int.natCast_emod,
    hlayer, hstate, add_zero] at hsetupSelect hsetupOut
  dsimp only [Stage_decrypt.parallel_sequential_generatedRoot_8_5] at hruntime
  rcases hruntime with ⟨runtimeSelected, _, _, _, hruntimeSelect, hruntimeOut⟩
  simp only [Int.ofNat_eq_natCast, add_zero] at hruntimeSelect
  rcases hsetupSelect with ⟨setupPosition, hsetupPosition, hsetupValue⟩
  rcases hruntimeSelect with ⟨runtimePosition, hruntimePosition, hruntimeValue⟩
  have hp : setupPosition = runtimePosition := by
    apply Fin.ext
    omega
  subst runtimePosition
  have hselected : setupSelected = runtimeSelected := hsetupValue.trans hruntimeValue.symm
  rw [hsetupOut, hselected, ← hruntimeOut, Nat.cast_mul]

#print axioms generated_source_index_agrees

theorem generated_source_pool_lookup
    (backend : BackendContext) (hashModel : HashModel) (params : Stage_encrypt.Params)
    (slot : Nat) (index : Int) (bases : Fin basePoolCount → ExactMatrix q n 2 inner)
    (trapdoors : Fin basePoolCount → TrapdoorValue (ExactMatrix q n 2 inner) Unit)
    (sourcePublic : ExactMatrix q n 2 inner)
    (sourceTrapdoor : TrapdoorValue (ExactMatrix q n 2 inner) Unit)
    (hrun : Stage_encrypt.parallel_generatedRoot_66 backend hashModel params slot
      (index, bases, trapdoors, ()) (sourcePublic, sourceTrapdoor, ())) :
    ∃ position : Fin basePoolCount, (position.val : Int) = index ∧ sourcePublic = bases position ∧
      sourceTrapdoor = trapdoors position := by
  rcases hrun with ⟨matrixValue, trapdoorValue, _, _, ⟨position, hposition, hmatrix⟩,
    _, _, ⟨trapdoorPosition, htrapdoorPosition, htrapdoor⟩, hout⟩
  have hp : position = trapdoorPosition := by
    apply Fin.ext
    omega
  subst trapdoorPosition
  refine ⟨position, hposition, ?_, ?_⟩
  · exact (congrArg Prod.fst hout).trans hmatrix
  · exact (congrArg (fun value ↦ value.2.1) hout).trans htrapdoor

theorem generated_target_pool_lookup
    (backend : BackendContext) (hashModel : HashModel) (params : Stage_encrypt.Params)
    (slot : Nat) (index : Int) (bases : Fin basePoolCount → ExactMatrix q n 2 inner)
    (targetPublic : ExactMatrix q n 2 inner)
    (hrun : Stage_encrypt.parallel_generatedRoot_71 backend hashModel params slot
      (index, bases, ()) targetPublic) :
    ∃ position : Fin basePoolCount, (position.val : Int) = index ∧ targetPublic = bases position := by
  rcases hrun with ⟨value, _, _, ⟨position, hposition, hvalue⟩, hout⟩
  exact ⟨position, hposition, hout.trans hvalue⟩

theorem generated_selected_transition_lookup
    (backend : BackendContext) (params : Stage_decrypt.Params) (layer lane : Nat)
    (index : Int) (transitions : Fin transitionCount → ExactMatrix q n inner inner)
    (selected : ExactMatrix q n inner inner)
    (hrun : Stage_decrypt.parallel_sequential_generatedRoot_8_12 backend params layer lane
      (index, transitions, ()) selected) :
    ∃ position : Fin transitionCount, (position.val : Int) = index ∧ selected = transitions position := by
  rcases hrun with ⟨value, _, _, ⟨position, hposition, hvalue⟩, hout⟩
  exact ⟨position, hposition, hout.trans hvalue⟩

#print axioms generated_source_pool_lookup

end DiamondGeneratedProof
