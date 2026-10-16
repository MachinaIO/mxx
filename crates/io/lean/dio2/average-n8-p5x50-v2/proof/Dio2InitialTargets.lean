import Dio2InitialRows
import Dio2Sharp

/-! Generated initial target and gain contracts. Public labels are monomials,
so their action gain is one independently of ring dimension. -/

set_option maxHeartbeats 4000000
set_option maxRecDepth 16384
set_option linter.unusedVariables false

attribute [local irreducible] Mxx.Primitives.negacyclicModulus

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut

theorem smulGain_message_one {n : Nat} (hn : 1 < n) (exponent : Int) :
    SmulGain (message (n := n) 1 exponent) 1 := by
  unfold message
  rw [Int.cast_one, one_mul]
  exact smulGain_root_pow hn (exponent % (2 * n : Nat)).toNat

/-- Imported public labels may have either sign; neither sign costs a dimension factor. -/
theorem smulGain_message_signed {n : Nat} (hn : 1 < n) (coefficient exponent : Int)
    (hsign : coefficient = 1 ∨ coefficient = -1) :
    SmulGain (message (n := n) coefficient exponent) 1 := by
  rcases hsign with rfl | rfl
  · exact smulGain_message_one hn exponent
  · have hnegative : message (n := n) (-1) exponent = -message 1 exponent := by
      simp only [message, Int.cast_neg, Int.cast_one, neg_one_mul, one_mul]
    rw [hnegative]
    intro rows columns error bound herror
    exact smulGain_neg (smulGain_message_one hn exponent) herror

open W_Stage_initial

/-- The `helper_Aut(11)` target in the public-action form used by the gain-aware cell lemmas. -/
theorem initial_rel_helper_Aut_11 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 1) :
    Rel sm (rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(11)]))] modulus_0 25)
      (fun port ↦ reducePoly modulus_0 8 ((1) * unitPair (autoPlaintext (((MxxIR.selectInt ((Int.ofNat (member.val))) [(11)]))) (sm)) port) • G)
      ![(outputs.2.1 member), (outputs.2.2.1 member)] 1 :=
  initial_row_helper_Aut_11 h hmask hpay hG member

theorem initial_gain_helper_Aut_11 {inputs : (Mxx.Primitives.ExactMatrix modulus_0 8 1 1) × (Mxx.Primitives.ExactMatrix modulus_0 8 2 54) × ByteArray × (Fin 5 → Int) × (Mxx.Primitives.ExactMatrix modulus_0 8 1 1) × Int × Unit}
    {sm tp : ErrorPoly 8} (member : Fin 1) : SmulGain (n := 8) (1) 1 := by
  intro rows columns error bound herror
  simpa only [one_smul, Nat.one_mul] using herror

/-- The `helper_Aut(13)` target in the public-action form used by the gain-aware cell lemmas. -/
theorem initial_rel_helper_Aut_13 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 1) :
    Rel sm (rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(13)]))] modulus_0 25)
      (fun port ↦ reducePoly modulus_0 8 ((1) * unitPair (autoPlaintext (((MxxIR.selectInt ((Int.ofNat (member.val))) [(13)]))) (sm)) port) • G)
      ![(outputs.2.2.2.1 member), (outputs.2.2.2.2.1 member)] 1 :=
  initial_row_helper_Aut_13 h hmask hpay hG member

theorem initial_gain_helper_Aut_13 {inputs : (Mxx.Primitives.ExactMatrix modulus_0 8 1 1) × (Mxx.Primitives.ExactMatrix modulus_0 8 2 54) × ByteArray × (Fin 5 → Int) × (Mxx.Primitives.ExactMatrix modulus_0 8 1 1) × Int × Unit}
    {sm tp : ErrorPoly 8} (member : Fin 1) : SmulGain (n := 8) (1) 1 := by
  intro rows columns error bound herror
  simpa only [one_smul, Nat.one_mul] using herror

/-- The `helper_Aut(15)` target in the public-action form used by the gain-aware cell lemmas. -/
theorem initial_rel_helper_Aut_15 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 1) :
    Rel sm (rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(15)]))] modulus_0 25)
      (fun port ↦ reducePoly modulus_0 8 ((1) * unitPair (autoPlaintext (((MxxIR.selectInt ((Int.ofNat (member.val))) [(15)]))) (sm)) port) • G)
      ![(outputs.2.2.2.2.2.1 member), (outputs.2.2.2.2.2.2.1 member)] 1 :=
  initial_row_helper_Aut_15 h hmask hpay hG member

theorem initial_gain_helper_Aut_15 {inputs : (Mxx.Primitives.ExactMatrix modulus_0 8 1 1) × (Mxx.Primitives.ExactMatrix modulus_0 8 2 54) × ByteArray × (Fin 5 → Int) × (Mxx.Primitives.ExactMatrix modulus_0 8 1 1) × Int × Unit}
    {sm tp : ErrorPoly 8} (member : Fin 1) : SmulGain (n := 8) (1) 1 := by
  intro rows columns error bound herror
  simpa only [one_smul, Nat.one_mul] using herror

/-- The `helper_Aut(3)` target in the public-action form used by the gain-aware cell lemmas. -/
theorem initial_rel_helper_Aut_3 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 1) :
    Rel sm (rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(3)]))] modulus_0 25)
      (fun port ↦ reducePoly modulus_0 8 ((1) * unitPair (autoPlaintext (((MxxIR.selectInt ((Int.ofNat (member.val))) [(3)]))) (sm)) port) • G)
      ![(outputs.2.2.2.2.2.2.2.1 member), (outputs.2.2.2.2.2.2.2.2.1 member)] 1 :=
  initial_row_helper_Aut_3 h hmask hpay hG member

theorem initial_gain_helper_Aut_3 {inputs : (Mxx.Primitives.ExactMatrix modulus_0 8 1 1) × (Mxx.Primitives.ExactMatrix modulus_0 8 2 54) × ByteArray × (Fin 5 → Int) × (Mxx.Primitives.ExactMatrix modulus_0 8 1 1) × Int × Unit}
    {sm tp : ErrorPoly 8} (member : Fin 1) : SmulGain (n := 8) (1) 1 := by
  intro rows columns error bound herror
  simpa only [one_smul, Nat.one_mul] using herror

/-- The `helper_Aut(5)` target in the public-action form used by the gain-aware cell lemmas. -/
theorem initial_rel_helper_Aut_5 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 1) :
    Rel sm (rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(5)]))] modulus_0 25)
      (fun port ↦ reducePoly modulus_0 8 ((1) * unitPair (autoPlaintext (((MxxIR.selectInt ((Int.ofNat (member.val))) [(5)]))) (sm)) port) • G)
      ![(outputs.2.2.2.2.2.2.2.2.2.1 member), (outputs.2.2.2.2.2.2.2.2.2.2.1 member)] 1 :=
  initial_row_helper_Aut_5 h hmask hpay hG member

theorem initial_gain_helper_Aut_5 {inputs : (Mxx.Primitives.ExactMatrix modulus_0 8 1 1) × (Mxx.Primitives.ExactMatrix modulus_0 8 2 54) × ByteArray × (Fin 5 → Int) × (Mxx.Primitives.ExactMatrix modulus_0 8 1 1) × Int × Unit}
    {sm tp : ErrorPoly 8} (member : Fin 1) : SmulGain (n := 8) (1) 1 := by
  intro rows columns error bound herror
  simpa only [one_smul, Nat.one_mul] using herror

/-- The `helper_Aut(7)` target in the public-action form used by the gain-aware cell lemmas. -/
theorem initial_rel_helper_Aut_7 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 1) :
    Rel sm (rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(7)]))] modulus_0 25)
      (fun port ↦ reducePoly modulus_0 8 ((1) * unitPair (autoPlaintext (((MxxIR.selectInt ((Int.ofNat (member.val))) [(7)]))) (sm)) port) • G)
      ![(outputs.2.2.2.2.2.2.2.2.2.2.2.1 member), (outputs.2.2.2.2.2.2.2.2.2.2.2.2.1 member)] 1 :=
  initial_row_helper_Aut_7 h hmask hpay hG member

theorem initial_gain_helper_Aut_7 {inputs : (Mxx.Primitives.ExactMatrix modulus_0 8 1 1) × (Mxx.Primitives.ExactMatrix modulus_0 8 2 54) × ByteArray × (Fin 5 → Int) × (Mxx.Primitives.ExactMatrix modulus_0 8 1 1) × Int × Unit}
    {sm tp : ErrorPoly 8} (member : Fin 1) : SmulGain (n := 8) (1) 1 := by
  intro rows columns error bound herror
  simpa only [one_smul, Nat.one_mul] using herror

/-- The `helper_Aut(9)` target in the public-action form used by the gain-aware cell lemmas. -/
theorem initial_rel_helper_Aut_9 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 1) :
    Rel sm (rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(9)]))] modulus_0 25)
      (fun port ↦ reducePoly modulus_0 8 ((1) * unitPair (autoPlaintext (((MxxIR.selectInt ((Int.ofNat (member.val))) [(9)]))) (sm)) port) • G)
      ![(outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1 member), (outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 member)] 1 :=
  initial_row_helper_Aut_9 h hmask hpay hG member

theorem initial_gain_helper_Aut_9 {inputs : (Mxx.Primitives.ExactMatrix modulus_0 8 1 1) × (Mxx.Primitives.ExactMatrix modulus_0 8 2 54) × ByteArray × (Fin 5 → Int) × (Mxx.Primitives.ExactMatrix modulus_0 8 1 1) × Int × Unit}
    {sm tp : ErrorPoly 8} (member : Fin 1) : SmulGain (n := 8) (1) 1 := by
  intro rows columns error bound herror
  simpa only [one_smul, Nat.one_mul] using herror

/-- The `helper_ExpLayout { stride_log2: 0, width_log2: 3 }` target in the public-action form used by the gain-aware cell lemmas. -/
theorem initial_rel_helper_ExpLayout_stride_log2_0_width_log2_3 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 8) :
    Rel sm (rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(1), (3), (5), (7), (9), (11), (13), (15)]))] modulus_0 25)
      (fun port ↦ reducePoly modulus_0 8 ((1) * unitPair (autoPlaintext (((MxxIR.selectInt ((Int.ofNat (member.val))) [(1), (3), (5), (7), (9), (11), (13), (15)]))) (sm)) port) • G)
      ![(outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 member), (outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 member)] 1 :=
  initial_row_helper_ExpLayout_stride_log2_0_width_log2_3 h hmask hpay hG member

theorem initial_gain_helper_ExpLayout_stride_log2_0_width_log2_3 {inputs : (Mxx.Primitives.ExactMatrix modulus_0 8 1 1) × (Mxx.Primitives.ExactMatrix modulus_0 8 2 54) × ByteArray × (Fin 5 → Int) × (Mxx.Primitives.ExactMatrix modulus_0 8 1 1) × Int × Unit}
    {sm tp : ErrorPoly 8} (member : Fin 8) : SmulGain (n := 8) (1) 1 := by
  intro rows columns error bound herror
  simpa only [one_smul, Nat.one_mul] using herror

/-- The `helper_ExpLayout { stride_log2: 1, width_log2: 2 }` target in the public-action form used by the gain-aware cell lemmas. -/
theorem initial_rel_helper_ExpLayout_stride_log2_1_width_log2_2 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 4) :
    Rel sm (rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(1), (3), (5), (15)]))] modulus_0 25)
      (fun port ↦ reducePoly modulus_0 8 ((1) * unitPair (autoPlaintext (((MxxIR.selectInt ((Int.ofNat (member.val))) [(1), (3), (5), (15)]))) (sm)) port) • G)
      ![(outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 member), (outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 member)] 1 :=
  initial_row_helper_ExpLayout_stride_log2_1_width_log2_2 h hmask hpay hG member

theorem initial_gain_helper_ExpLayout_stride_log2_1_width_log2_2 {inputs : (Mxx.Primitives.ExactMatrix modulus_0 8 1 1) × (Mxx.Primitives.ExactMatrix modulus_0 8 2 54) × ByteArray × (Fin 5 → Int) × (Mxx.Primitives.ExactMatrix modulus_0 8 1 1) × Int × Unit}
    {sm tp : ErrorPoly 8} (member : Fin 4) : SmulGain (n := 8) (1) 1 := by
  intro rows columns error bound herror
  simpa only [one_smul, Nat.one_mul] using herror

/-- The `row_dio2/label-0` target in the public-action form used by the gain-aware cell lemmas. -/
theorem initial_rel_row_dio2_label_0 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 2) :
    Rel sm (rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48] [.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat (member.val)))] modulus_0 25)
      (fun port ↦ reducePoly modulus_0 8 ((message 1 (MxxIR.selectInt (if (decide (((Int.ofNat (member.val))) = (1))) then 1 else 0) [((intFamilyValue (((inputs.2.2.2.1))) ((((((Int.ofNat (member.val))) % (1))) * (5)))) * (1)), ((0) + (0))])) * unitPair (sm) port) • G)
      ![(outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 member), (outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 member)] 1 :=
  initial_row_row_dio2_label_0 h hmask hpay hG member

theorem initial_gain_row_dio2_label_0 {inputs : (Mxx.Primitives.ExactMatrix modulus_0 8 1 1) × (Mxx.Primitives.ExactMatrix modulus_0 8 2 54) × ByteArray × (Fin 5 → Int) × (Mxx.Primitives.ExactMatrix modulus_0 8 1 1) × Int × Unit}
    {sm tp : ErrorPoly 8} (member : Fin 2) : SmulGain (n := 8) (message 1 (MxxIR.selectInt (if (decide (((Int.ofNat (member.val))) = (1))) then 1 else 0) [((intFamilyValue (((inputs.2.2.2.1))) ((((((Int.ofNat (member.val))) % (1))) * (5)))) * (1)), ((0) + (0))])) 1 := by
  exact smulGain_message_one (by decide) _

/-- The `row_dio2/label` target in the public-action form used by the gain-aware cell lemmas. -/
theorem initial_rel_row_dio2_label {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 4) :
    Rel sm (rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108] [.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat (member.val)))] modulus_0 25)
      (fun port ↦ reducePoly modulus_0 8 ((message 1 (((intFamilyValue (((inputs.2.2.2.1))) ((((((((((Int.ofNat (member.val))) / (4))) * (5))) + ((((Int.ofNat (member.val))) % (4))))) + (1)))) * (1)))) * unitPair (sm) port) • G)
      ![(outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 member), (outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 member)] 1 :=
  initial_row_row_dio2_label h hmask hpay hG member

theorem initial_gain_row_dio2_label {inputs : (Mxx.Primitives.ExactMatrix modulus_0 8 1 1) × (Mxx.Primitives.ExactMatrix modulus_0 8 2 54) × ByteArray × (Fin 5 → Int) × (Mxx.Primitives.ExactMatrix modulus_0 8 1 1) × Int × Unit}
    {sm tp : ErrorPoly 8} (member : Fin 4) : SmulGain (n := 8) (message 1 (((intFamilyValue (((inputs.2.2.2.1))) ((((((((((Int.ofNat (member.val))) / (4))) * (5))) + ((((Int.ofNat (member.val))) % (4))))) + (1)))) * (1)))) 1 := by
  exact smulGain_message_one (by decide) _

/-- The `row_dio2/one-private` target in the public-action form used by the gain-aware cell lemmas. -/
theorem initial_rel_row_dio2_one_private {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) :
    Rel sm (rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101] [] modulus_0 25)
      (fun port ↦ reducePoly modulus_0 8 ((1) * unitPair (tp) port) • G)
      ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1] 1 :=
  initial_row_row_dio2_one_private h hmask hpay hG

theorem initial_gain_row_dio2_one_private {inputs : (Mxx.Primitives.ExactMatrix modulus_0 8 1 1) × (Mxx.Primitives.ExactMatrix modulus_0 8 2 54) × ByteArray × (Fin 5 → Int) × (Mxx.Primitives.ExactMatrix modulus_0 8 1 1) × Int × Unit}
    {sm tp : ErrorPoly 8} : SmulGain (n := 8) (1) 1 := by
  intro rows columns error bound herror
  simpa only [one_smul, Nat.one_mul] using herror

/-- The `row_dio2/state` target in the public-action form used by the gain-aware cell lemmas. -/
theorem initial_rel_row_dio2_state {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) :
    Rel sm (rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101] [] modulus_0 25)
      (fun port ↦ reducePoly modulus_0 8 ((message 1 (((inputs.2.2.2.2.2.1) * (2)))) * unitPair (tp) port) • G)
      ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1] 1 :=
  initial_row_row_dio2_state h hmask hpay hG

theorem initial_gain_row_dio2_state {inputs : (Mxx.Primitives.ExactMatrix modulus_0 8 1 1) × (Mxx.Primitives.ExactMatrix modulus_0 8 2 54) × ByteArray × (Fin 5 → Int) × (Mxx.Primitives.ExactMatrix modulus_0 8 1 1) × Int × Unit}
    {sm tp : ErrorPoly 8} : SmulGain (n := 8) (message 1 (((inputs.2.2.2.2.2.1) * (2)))) 1 := by
  exact smulGain_message_one (by decide) _

end Dio2Proof
