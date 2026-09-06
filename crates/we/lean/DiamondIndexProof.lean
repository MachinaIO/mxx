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
    (params : Stage_encrypt.Params)
    (stateCount digitBase layer : Nat) (hstates : 0 < stateCount) (hdigits : 0 < digitBase)
    (hstateGeometry : (stateCount : Int) =
      1 + params.diamond_batch_bits * params.diamond_input_count)
    (hbaseGeometry : (digitBase : Int) = params.diamond_digit_base)
    (digit : Fin digitBase) (state : Fin stateCount) (index : Int)
    (hrun : index =
      ((((layer * digitBase + digit.val) * stateCount + state.val : Nat) : Int) /
        (params.diamond_batch_bits * params.diamond_digit_base * params.diamond_input_count + params.diamond_digit_base) + 1) *
        (1 + params.diamond_batch_bits * params.diamond_input_count) +
      (((layer * digitBase + digit.val) * stateCount + state.val : Nat) : Int) %
        (1 + params.diamond_batch_bits * params.diamond_input_count)) :
    index = ((layer + 1) * stateCount + state.val : Nat) := by
  have hwidth : ((digitBase * stateCount : Nat) : Int) =
      params.diamond_batch_bits * params.diamond_digit_base * params.diamond_input_count +
        params.diamond_digit_base := by
    rw [Nat.cast_mul, hstateGeometry, hbaseGeometry]
    ring
  have heq := hrun
  rw [← hwidth, ← hstateGeometry] at heq
  obtain ⟨hlayer, hstate, _⟩ := transition_coordinates stateCount digitBase layer
    hstates hdigits digit state
  simp only [← Int.natCast_ediv, ← Int.natCast_emod,
    hlayer, hstate] at heq
  simpa only [Nat.cast_add, Nat.cast_mul, Nat.cast_one] using heq

theorem generated_runtime_transition_index
    (params : Stage_decrypt.Params)
    (stateCount digitBase layer digit state : Nat)
    (hstateGeometry : (stateCount : Int) =
      1 + params.diamond_batch_bits * params.diamond_input_count)
    (hbaseGeometry : (digitBase : Int) = params.diamond_digit_base)
    (index : Int)
    (hrun : index = (layer : Int) *
      (params.diamond_batch_bits * params.diamond_digit_base * params.diamond_input_count +
        params.diamond_digit_base) + (digit : Int) *
      (1 + params.diamond_batch_bits * params.diamond_input_count) + (state : Int)) :
    index = ((layer * digitBase + digit) * stateCount + state : Nat) := by
  have hwidth : ((digitBase * stateCount : Nat) : Int) =
      params.diamond_batch_bits * params.diamond_digit_base * params.diamond_input_count +
        params.diamond_digit_base := by
    rw [Nat.cast_mul, hstateGeometry, hbaseGeometry]
    ring
  rw [← hwidth, ← hstateGeometry] at hrun
  convert hrun using 1
  simp only [Nat.cast_add, Nat.cast_mul]
  ring

#print axioms generated_runtime_transition_index

theorem generated_source_index_agrees
    (params : Stage_encrypt.Params)
    (stateCount digitBase layer : Nat) (hstates : 0 < stateCount) (hdigits : 0 < digitBase)
    (hstateGeometry : (stateCount : Int) =
      1 + params.diamond_batch_bits * params.diamond_input_count)
    (hbaseGeometry : (digitBase : Int) = params.diamond_digit_base)
    (digit : Fin digitBase) (state : Fin stateCount) (setupIndex sourceIndex : Int)
    (hsetup : ∃ selected : Int,
      select (if decide (((((layer * digitBase + digit.val) * stateCount + state.val : Nat) : Int) /
          (params.diamond_batch_bits * params.diamond_digit_base * params.diamond_input_count +
            params.diamond_digit_base)) * params.diamond_batch_bits + 1 ≤
          (((layer * digitBase + digit.val) * stateCount + state.val : Nat) : Int) %
            (1 + params.diamond_batch_bits * params.diamond_input_count)) then 1 else 0)
        [(((layer * digitBase + digit.val) * stateCount + state.val : Nat) : Int) %
          (1 + params.diamond_batch_bits * params.diamond_input_count), 0] selected ∧
      setupIndex = ((((layer * digitBase + digit.val) * stateCount + state.val : Nat) : Int) /
        (params.diamond_batch_bits * params.diamond_digit_base * params.diamond_input_count +
          params.diamond_digit_base)) * (1 + params.diamond_batch_bits * params.diamond_input_count) + selected)
    (hruntime : select (if decide ((layer : Int) * params.diamond_batch_bits + 1 ≤ (state.val : Int)) then 1 else 0)
      [(state.val : Int), 0] sourceIndex) :
    setupIndex = (layer * stateCount : Nat) + sourceIndex := by
  have hwidth : ((digitBase * stateCount : Nat) : Int) =
      params.diamond_batch_bits * params.diamond_digit_base * params.diamond_input_count +
        params.diamond_digit_base := by
    rw [Nat.cast_mul, hstateGeometry, hbaseGeometry]
    ring
  obtain ⟨hlayer, hstate, _⟩ := transition_coordinates stateCount digitBase layer
    hstates hdigits digit state
  rcases hsetup with ⟨setupSelected, hsetupSelect, hsetupOut⟩
  rw [← hwidth, ← hstateGeometry] at hsetupSelect hsetupOut
  simp only [← Int.natCast_ediv, ← Int.natCast_emod,
    hlayer, hstate] at hsetupSelect hsetupOut
  have hruntimeSelect := hruntime
  rcases hsetupSelect with ⟨setupPosition, hsetupPosition, hsetupValue⟩
  rcases hruntimeSelect with ⟨runtimePosition, hruntimePosition, hruntimeValue⟩
  have hp : setupPosition = runtimePosition := by
    apply Fin.ext
    omega
  subst runtimePosition
  have hselected : setupSelected = sourceIndex := hsetupValue.trans hruntimeValue.symm
  rw [hsetupOut, hselected, Nat.cast_mul]

#print axioms generated_source_index_agrees

theorem generated_source_pool_lookup
    (index : Int) (bases : Fin basePoolCount → ExactMatrix q n 2 inner)
    (trapdoors : Fin basePoolCount → TrapdoorValue (ExactMatrix q n 2 inner) Unit)
    (sourcePublic : ExactMatrix q n 2 inner)
    (sourceTrapdoor : TrapdoorValue (ExactMatrix q n 2 inner) Unit)
    (hrun : familyGetDynamic bases index sourcePublic ∧ familyGetDynamic trapdoors index sourceTrapdoor) :
    ∃ position : Fin basePoolCount, (position.val : Int) = index ∧ sourcePublic = bases position ∧
      sourceTrapdoor = trapdoors position := by
  rcases hrun with ⟨⟨position, hposition, hmatrix⟩,
    ⟨trapdoorPosition, htrapdoorPosition, htrapdoor⟩⟩
  have hp : position = trapdoorPosition := by
    apply Fin.ext
    omega
  subst trapdoorPosition
  refine ⟨position, hposition, ?_, ?_⟩
  · exact hmatrix
  · exact htrapdoor

theorem generated_target_pool_lookup
    (index : Int) (bases : Fin basePoolCount → ExactMatrix q n 2 inner)
    (targetPublic : ExactMatrix q n 2 inner)
    (hrun : familyGetDynamic bases index targetPublic) :
    ∃ position : Fin basePoolCount, (position.val : Int) = index ∧ targetPublic = bases position := by
  exact hrun

theorem generated_selected_transition_lookup
    (index : Int) (transitions : Fin transitionCount → ExactMatrix q n inner inner)
    (selected : ExactMatrix q n inner inner)
    (hrun : familyGetDynamic transitions index selected) :
    ∃ position : Fin transitionCount, (position.val : Int) = index ∧ selected = transitions position := by
  exact hrun

#print axioms generated_source_pool_lookup

end DiamondGeneratedProof
