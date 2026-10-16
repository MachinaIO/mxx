import Dio2Initial
import W_Stage_initial_Facts

/-! Generated family-level initial encodings. All member indices remain quantified. -/

set_option maxHeartbeats 4000000
set_option maxRecDepth 16384
set_option linter.unusedVariables false

attribute [local irreducible] Mxx.Primitives.negacyclicModulus

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut

open W_Stage_initial

theorem initial_component_helper_Aut_11_0 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 1) :
    RootComponent model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(11)]))] sm G
      ((1) * (autoPlaintext (((MxxIR.selectInt ((Int.ofNat (member.val))) [(11)]))) (sm))) (outputs.2.1 member) 0 := by
  obtain ⟨witness, hbody⟩ := h
  let ctx : generatedRoot.Ctx backend model params := ⟨inputs, witness⟩
  have F := generatedRoot.facts ctx hbody
  rw [F.c2.hout]
  have hscope := F.c0.h10_0 member
  change parallel_generatedRoot_10 backend model params member.val ((inputs.1), (inputs.2.2.1), ((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)), ()) _ at hscope
  obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, constraints⟩ := hscope
  have hunit : (((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1))) = 1 := rfl
  have P_w_0_0 : ((inputs.1)) 0 0 = reducePoly modulus_0 8 (sm) := hmask
  have P_w_9_0 : w_9_0 0 0 = reducePoly modulus_0 8 (autoPlaintext (((MxxIR.selectInt ((Int.ofNat member.val)) [(11)]))) (sm)) := auto_plaintext_entry (by decide) (by decide) constraints.2.2.2.2.2.2.1 P_w_0_0
  have key0 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(11)]))]
      modulus_0 25 0 0 = w_3_0 := by
    rw [hashSample_eq constraints.2.2.1]
    rfl
  have key1 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(11)]))]
      modulus_0 25 1 0 = w_6_0 := by
    rw [hashSample_eq constraints.2.2.2.1]
    rfl
  have gadgetEq : w_10_0 = G :=
    gadget_matrix_unique constraints.2.2.2.2.2.2.2.1 hG
  unfold RootComponent
  rw [key0, key1, ← gadgetEq]
  apply component_encode hmask (by rfl)
    (by simpa only [one_mul, mul_one] using P_w_9_0)
    constraints.2.2.2.2.2.2.2.2.1 constraints.2.2.2.2.2.2.2.2.2.1
  simpa only [mul_one, one_mul] using constraints.2.2.2.2.2.2.2.2.2.2

theorem initial_component_helper_Aut_11_1 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 1) :
    RootComponent model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(11)]))] sm G
      (1) (outputs.2.2.1 member) 1 := by
  obtain ⟨witness, hbody⟩ := h
  let ctx : generatedRoot.Ctx backend model params := ⟨inputs, witness⟩
  have F := generatedRoot.facts ctx hbody
  rw [F.c2.hout]
  have hscope := F.c0.h11_0 member
  change parallel_generatedRoot_11 backend model params member.val ((inputs.1), (inputs.2.2.1), ((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)), ()) _ at hscope
  obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, constraints⟩ := hscope
  have hunit : (((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1))) = 1 := rfl
  have P_w_9_0 : ((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)) 0 0 = reducePoly modulus_0 8 (1) := one_entry
  have key0 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(11)]))]
      modulus_0 25 0 1 = w_3_0 := by
    rw [hashSample_eq constraints.2.2.1]
    rfl
  have key1 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(11)]))]
      modulus_0 25 1 1 = w_6_0 := by
    rw [hashSample_eq constraints.2.2.2.1]
    rfl
  have gadgetEq : w_10_0 = G :=
    gadget_matrix_unique constraints.2.2.2.2.1 hG
  unfold RootComponent
  rw [key0, key1, ← gadgetEq]
  apply component_encode hmask (by rfl)
    (by simpa only [one_mul, mul_one] using P_w_9_0)
    constraints.2.2.2.2.2.1 constraints.2.2.2.2.2.2.1
  simpa only [mul_one, one_mul] using constraints.2.2.2.2.2.2.2

/-- Every member of `helper_Aut(11)` is a root encoding with coefficient error at most one. -/
theorem initial_row_helper_Aut_11 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 1) :
    RootRow model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(11)]))] sm G
      (fun port ↦ fun _ _ ↦ reducePoly modulus_0 8
        ((1) * unitPair (autoPlaintext (((MxxIR.selectInt ((Int.ofNat (member.val))) [(11)]))) (sm)) port))
      ![(outputs.2.1 member), (outputs.2.2.1 member)] := by
  apply rootRow_pair_unit
  · simpa only [one_mul, mul_one] using initial_component_helper_Aut_11_0 h hmask hpay hG member
  · simpa only [one_mul, mul_one] using initial_component_helper_Aut_11_1 h hmask hpay hG member

theorem initial_component_helper_Aut_13_0 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 1) :
    RootComponent model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(13)]))] sm G
      ((1) * (autoPlaintext (((MxxIR.selectInt ((Int.ofNat (member.val))) [(13)]))) (sm))) (outputs.2.2.2.1 member) 0 := by
  obtain ⟨witness, hbody⟩ := h
  let ctx : generatedRoot.Ctx backend model params := ⟨inputs, witness⟩
  have F := generatedRoot.facts ctx hbody
  rw [F.c2.hout]
  have hscope := F.c0.h12_0 member
  change parallel_generatedRoot_12 backend model params member.val ((inputs.1), (inputs.2.2.1), ((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)), ()) _ at hscope
  obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, constraints⟩ := hscope
  have hunit : (((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1))) = 1 := rfl
  have P_w_0_0 : ((inputs.1)) 0 0 = reducePoly modulus_0 8 (sm) := hmask
  have P_w_9_0 : w_9_0 0 0 = reducePoly modulus_0 8 (autoPlaintext (((MxxIR.selectInt ((Int.ofNat member.val)) [(13)]))) (sm)) := auto_plaintext_entry (by decide) (by decide) constraints.2.2.2.2.2.2.1 P_w_0_0
  have key0 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(13)]))]
      modulus_0 25 0 0 = w_3_0 := by
    rw [hashSample_eq constraints.2.2.1]
    rfl
  have key1 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(13)]))]
      modulus_0 25 1 0 = w_6_0 := by
    rw [hashSample_eq constraints.2.2.2.1]
    rfl
  have gadgetEq : w_10_0 = G :=
    gadget_matrix_unique constraints.2.2.2.2.2.2.2.1 hG
  unfold RootComponent
  rw [key0, key1, ← gadgetEq]
  apply component_encode hmask (by rfl)
    (by simpa only [one_mul, mul_one] using P_w_9_0)
    constraints.2.2.2.2.2.2.2.2.1 constraints.2.2.2.2.2.2.2.2.2.1
  simpa only [mul_one, one_mul] using constraints.2.2.2.2.2.2.2.2.2.2

theorem initial_component_helper_Aut_13_1 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 1) :
    RootComponent model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(13)]))] sm G
      (1) (outputs.2.2.2.2.1 member) 1 := by
  obtain ⟨witness, hbody⟩ := h
  let ctx : generatedRoot.Ctx backend model params := ⟨inputs, witness⟩
  have F := generatedRoot.facts ctx hbody
  rw [F.c2.hout]
  have hscope := F.c0.h13_0 member
  change parallel_generatedRoot_13 backend model params member.val ((inputs.1), (inputs.2.2.1), ((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)), ()) _ at hscope
  obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, constraints⟩ := hscope
  have hunit : (((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1))) = 1 := rfl
  have P_w_9_0 : ((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)) 0 0 = reducePoly modulus_0 8 (1) := one_entry
  have key0 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(13)]))]
      modulus_0 25 0 1 = w_3_0 := by
    rw [hashSample_eq constraints.2.2.1]
    rfl
  have key1 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(13)]))]
      modulus_0 25 1 1 = w_6_0 := by
    rw [hashSample_eq constraints.2.2.2.1]
    rfl
  have gadgetEq : w_10_0 = G :=
    gadget_matrix_unique constraints.2.2.2.2.1 hG
  unfold RootComponent
  rw [key0, key1, ← gadgetEq]
  apply component_encode hmask (by rfl)
    (by simpa only [one_mul, mul_one] using P_w_9_0)
    constraints.2.2.2.2.2.1 constraints.2.2.2.2.2.2.1
  simpa only [mul_one, one_mul] using constraints.2.2.2.2.2.2.2

/-- Every member of `helper_Aut(13)` is a root encoding with coefficient error at most one. -/
theorem initial_row_helper_Aut_13 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 1) :
    RootRow model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(13)]))] sm G
      (fun port ↦ fun _ _ ↦ reducePoly modulus_0 8
        ((1) * unitPair (autoPlaintext (((MxxIR.selectInt ((Int.ofNat (member.val))) [(13)]))) (sm)) port))
      ![(outputs.2.2.2.1 member), (outputs.2.2.2.2.1 member)] := by
  apply rootRow_pair_unit
  · simpa only [one_mul, mul_one] using initial_component_helper_Aut_13_0 h hmask hpay hG member
  · simpa only [one_mul, mul_one] using initial_component_helper_Aut_13_1 h hmask hpay hG member

theorem initial_component_helper_Aut_15_0 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 1) :
    RootComponent model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(15)]))] sm G
      ((1) * (autoPlaintext (((MxxIR.selectInt ((Int.ofNat (member.val))) [(15)]))) (sm))) (outputs.2.2.2.2.2.1 member) 0 := by
  obtain ⟨witness, hbody⟩ := h
  let ctx : generatedRoot.Ctx backend model params := ⟨inputs, witness⟩
  have F := generatedRoot.facts ctx hbody
  rw [F.c2.hout]
  have hscope := F.c0.h14_0 member
  change parallel_generatedRoot_14 backend model params member.val ((inputs.1), (inputs.2.2.1), ((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)), ()) _ at hscope
  obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, constraints⟩ := hscope
  have hunit : (((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1))) = 1 := rfl
  have P_w_0_0 : ((inputs.1)) 0 0 = reducePoly modulus_0 8 (sm) := hmask
  have P_w_9_0 : w_9_0 0 0 = reducePoly modulus_0 8 (autoPlaintext (((MxxIR.selectInt ((Int.ofNat member.val)) [(15)]))) (sm)) := auto_plaintext_entry (by decide) (by decide) constraints.2.2.2.2.2.2.1 P_w_0_0
  have key0 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(15)]))]
      modulus_0 25 0 0 = w_3_0 := by
    rw [hashSample_eq constraints.2.2.1]
    rfl
  have key1 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(15)]))]
      modulus_0 25 1 0 = w_6_0 := by
    rw [hashSample_eq constraints.2.2.2.1]
    rfl
  have gadgetEq : w_10_0 = G :=
    gadget_matrix_unique constraints.2.2.2.2.2.2.2.1 hG
  unfold RootComponent
  rw [key0, key1, ← gadgetEq]
  apply component_encode hmask (by rfl)
    (by simpa only [one_mul, mul_one] using P_w_9_0)
    constraints.2.2.2.2.2.2.2.2.1 constraints.2.2.2.2.2.2.2.2.2.1
  simpa only [mul_one, one_mul] using constraints.2.2.2.2.2.2.2.2.2.2

theorem initial_component_helper_Aut_15_1 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 1) :
    RootComponent model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(15)]))] sm G
      (1) (outputs.2.2.2.2.2.2.1 member) 1 := by
  obtain ⟨witness, hbody⟩ := h
  let ctx : generatedRoot.Ctx backend model params := ⟨inputs, witness⟩
  have F := generatedRoot.facts ctx hbody
  rw [F.c2.hout]
  have hscope := F.c0.h15_0 member
  change parallel_generatedRoot_15 backend model params member.val ((inputs.1), (inputs.2.2.1), ((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)), ()) _ at hscope
  obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, constraints⟩ := hscope
  have hunit : (((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1))) = 1 := rfl
  have P_w_9_0 : ((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)) 0 0 = reducePoly modulus_0 8 (1) := one_entry
  have key0 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(15)]))]
      modulus_0 25 0 1 = w_3_0 := by
    rw [hashSample_eq constraints.2.2.1]
    rfl
  have key1 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(15)]))]
      modulus_0 25 1 1 = w_6_0 := by
    rw [hashSample_eq constraints.2.2.2.1]
    rfl
  have gadgetEq : w_10_0 = G :=
    gadget_matrix_unique constraints.2.2.2.2.1 hG
  unfold RootComponent
  rw [key0, key1, ← gadgetEq]
  apply component_encode hmask (by rfl)
    (by simpa only [one_mul, mul_one] using P_w_9_0)
    constraints.2.2.2.2.2.1 constraints.2.2.2.2.2.2.1
  simpa only [mul_one, one_mul] using constraints.2.2.2.2.2.2.2

/-- Every member of `helper_Aut(15)` is a root encoding with coefficient error at most one. -/
theorem initial_row_helper_Aut_15 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 1) :
    RootRow model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(15)]))] sm G
      (fun port ↦ fun _ _ ↦ reducePoly modulus_0 8
        ((1) * unitPair (autoPlaintext (((MxxIR.selectInt ((Int.ofNat (member.val))) [(15)]))) (sm)) port))
      ![(outputs.2.2.2.2.2.1 member), (outputs.2.2.2.2.2.2.1 member)] := by
  apply rootRow_pair_unit
  · simpa only [one_mul, mul_one] using initial_component_helper_Aut_15_0 h hmask hpay hG member
  · simpa only [one_mul, mul_one] using initial_component_helper_Aut_15_1 h hmask hpay hG member

theorem initial_component_helper_Aut_3_0 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 1) :
    RootComponent model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(3)]))] sm G
      ((1) * (autoPlaintext (((MxxIR.selectInt ((Int.ofNat (member.val))) [(3)]))) (sm))) (outputs.2.2.2.2.2.2.2.1 member) 0 := by
  obtain ⟨witness, hbody⟩ := h
  let ctx : generatedRoot.Ctx backend model params := ⟨inputs, witness⟩
  have F := generatedRoot.facts ctx hbody
  rw [F.c2.hout]
  have hscope := F.c0.h16_0 member
  change parallel_generatedRoot_16 backend model params member.val ((inputs.1), (inputs.2.2.1), ((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)), ()) _ at hscope
  obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, constraints⟩ := hscope
  have hunit : (((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1))) = 1 := rfl
  have P_w_0_0 : ((inputs.1)) 0 0 = reducePoly modulus_0 8 (sm) := hmask
  have P_w_9_0 : w_9_0 0 0 = reducePoly modulus_0 8 (autoPlaintext (((MxxIR.selectInt ((Int.ofNat member.val)) [(3)]))) (sm)) := auto_plaintext_entry (by decide) (by decide) constraints.2.2.2.2.2.2.1 P_w_0_0
  have key0 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(3)]))]
      modulus_0 25 0 0 = w_3_0 := by
    rw [hashSample_eq constraints.2.2.1]
    rfl
  have key1 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(3)]))]
      modulus_0 25 1 0 = w_6_0 := by
    rw [hashSample_eq constraints.2.2.2.1]
    rfl
  have gadgetEq : w_10_0 = G :=
    gadget_matrix_unique constraints.2.2.2.2.2.2.2.1 hG
  unfold RootComponent
  rw [key0, key1, ← gadgetEq]
  apply component_encode hmask (by rfl)
    (by simpa only [one_mul, mul_one] using P_w_9_0)
    constraints.2.2.2.2.2.2.2.2.1 constraints.2.2.2.2.2.2.2.2.2.1
  simpa only [mul_one, one_mul] using constraints.2.2.2.2.2.2.2.2.2.2

theorem initial_component_helper_Aut_3_1 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 1) :
    RootComponent model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(3)]))] sm G
      (1) (outputs.2.2.2.2.2.2.2.2.1 member) 1 := by
  obtain ⟨witness, hbody⟩ := h
  let ctx : generatedRoot.Ctx backend model params := ⟨inputs, witness⟩
  have F := generatedRoot.facts ctx hbody
  rw [F.c2.hout]
  have hscope := F.c0.h17_0 member
  change parallel_generatedRoot_17 backend model params member.val ((inputs.1), (inputs.2.2.1), ((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)), ()) _ at hscope
  obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, constraints⟩ := hscope
  have hunit : (((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1))) = 1 := rfl
  have P_w_9_0 : ((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)) 0 0 = reducePoly modulus_0 8 (1) := one_entry
  have key0 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(3)]))]
      modulus_0 25 0 1 = w_3_0 := by
    rw [hashSample_eq constraints.2.2.1]
    rfl
  have key1 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(3)]))]
      modulus_0 25 1 1 = w_6_0 := by
    rw [hashSample_eq constraints.2.2.2.1]
    rfl
  have gadgetEq : w_10_0 = G :=
    gadget_matrix_unique constraints.2.2.2.2.1 hG
  unfold RootComponent
  rw [key0, key1, ← gadgetEq]
  apply component_encode hmask (by rfl)
    (by simpa only [one_mul, mul_one] using P_w_9_0)
    constraints.2.2.2.2.2.1 constraints.2.2.2.2.2.2.1
  simpa only [mul_one, one_mul] using constraints.2.2.2.2.2.2.2

/-- Every member of `helper_Aut(3)` is a root encoding with coefficient error at most one. -/
theorem initial_row_helper_Aut_3 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 1) :
    RootRow model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(3)]))] sm G
      (fun port ↦ fun _ _ ↦ reducePoly modulus_0 8
        ((1) * unitPair (autoPlaintext (((MxxIR.selectInt ((Int.ofNat (member.val))) [(3)]))) (sm)) port))
      ![(outputs.2.2.2.2.2.2.2.1 member), (outputs.2.2.2.2.2.2.2.2.1 member)] := by
  apply rootRow_pair_unit
  · simpa only [one_mul, mul_one] using initial_component_helper_Aut_3_0 h hmask hpay hG member
  · simpa only [one_mul, mul_one] using initial_component_helper_Aut_3_1 h hmask hpay hG member

theorem initial_component_helper_Aut_5_0 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 1) :
    RootComponent model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(5)]))] sm G
      ((1) * (autoPlaintext (((MxxIR.selectInt ((Int.ofNat (member.val))) [(5)]))) (sm))) (outputs.2.2.2.2.2.2.2.2.2.1 member) 0 := by
  obtain ⟨witness, hbody⟩ := h
  let ctx : generatedRoot.Ctx backend model params := ⟨inputs, witness⟩
  have F := generatedRoot.facts ctx hbody
  rw [F.c2.hout]
  have hscope := F.c0.h18_0 member
  change parallel_generatedRoot_18 backend model params member.val ((inputs.1), (inputs.2.2.1), ((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)), ()) _ at hscope
  obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, constraints⟩ := hscope
  have hunit : (((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1))) = 1 := rfl
  have P_w_0_0 : ((inputs.1)) 0 0 = reducePoly modulus_0 8 (sm) := hmask
  have P_w_9_0 : w_9_0 0 0 = reducePoly modulus_0 8 (autoPlaintext (((MxxIR.selectInt ((Int.ofNat member.val)) [(5)]))) (sm)) := auto_plaintext_entry (by decide) (by decide) constraints.2.2.2.2.2.2.1 P_w_0_0
  have key0 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(5)]))]
      modulus_0 25 0 0 = w_3_0 := by
    rw [hashSample_eq constraints.2.2.1]
    rfl
  have key1 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(5)]))]
      modulus_0 25 1 0 = w_6_0 := by
    rw [hashSample_eq constraints.2.2.2.1]
    rfl
  have gadgetEq : w_10_0 = G :=
    gadget_matrix_unique constraints.2.2.2.2.2.2.2.1 hG
  unfold RootComponent
  rw [key0, key1, ← gadgetEq]
  apply component_encode hmask (by rfl)
    (by simpa only [one_mul, mul_one] using P_w_9_0)
    constraints.2.2.2.2.2.2.2.2.1 constraints.2.2.2.2.2.2.2.2.2.1
  simpa only [mul_one, one_mul] using constraints.2.2.2.2.2.2.2.2.2.2

theorem initial_component_helper_Aut_5_1 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 1) :
    RootComponent model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(5)]))] sm G
      (1) (outputs.2.2.2.2.2.2.2.2.2.2.1 member) 1 := by
  obtain ⟨witness, hbody⟩ := h
  let ctx : generatedRoot.Ctx backend model params := ⟨inputs, witness⟩
  have F := generatedRoot.facts ctx hbody
  rw [F.c2.hout]
  have hscope := F.c0.h19_0 member
  change parallel_generatedRoot_19 backend model params member.val ((inputs.1), (inputs.2.2.1), ((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)), ()) _ at hscope
  obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, constraints⟩ := hscope
  have hunit : (((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1))) = 1 := rfl
  have P_w_9_0 : ((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)) 0 0 = reducePoly modulus_0 8 (1) := one_entry
  have key0 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(5)]))]
      modulus_0 25 0 1 = w_3_0 := by
    rw [hashSample_eq constraints.2.2.1]
    rfl
  have key1 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(5)]))]
      modulus_0 25 1 1 = w_6_0 := by
    rw [hashSample_eq constraints.2.2.2.1]
    rfl
  have gadgetEq : w_10_0 = G :=
    gadget_matrix_unique constraints.2.2.2.2.1 hG
  unfold RootComponent
  rw [key0, key1, ← gadgetEq]
  apply component_encode hmask (by rfl)
    (by simpa only [one_mul, mul_one] using P_w_9_0)
    constraints.2.2.2.2.2.1 constraints.2.2.2.2.2.2.1
  simpa only [mul_one, one_mul] using constraints.2.2.2.2.2.2.2

/-- Every member of `helper_Aut(5)` is a root encoding with coefficient error at most one. -/
theorem initial_row_helper_Aut_5 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 1) :
    RootRow model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(5)]))] sm G
      (fun port ↦ fun _ _ ↦ reducePoly modulus_0 8
        ((1) * unitPair (autoPlaintext (((MxxIR.selectInt ((Int.ofNat (member.val))) [(5)]))) (sm)) port))
      ![(outputs.2.2.2.2.2.2.2.2.2.1 member), (outputs.2.2.2.2.2.2.2.2.2.2.1 member)] := by
  apply rootRow_pair_unit
  · simpa only [one_mul, mul_one] using initial_component_helper_Aut_5_0 h hmask hpay hG member
  · simpa only [one_mul, mul_one] using initial_component_helper_Aut_5_1 h hmask hpay hG member

theorem initial_component_helper_Aut_7_0 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 1) :
    RootComponent model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(7)]))] sm G
      ((1) * (autoPlaintext (((MxxIR.selectInt ((Int.ofNat (member.val))) [(7)]))) (sm))) (outputs.2.2.2.2.2.2.2.2.2.2.2.1 member) 0 := by
  obtain ⟨witness, hbody⟩ := h
  let ctx : generatedRoot.Ctx backend model params := ⟨inputs, witness⟩
  have F := generatedRoot.facts ctx hbody
  rw [F.c2.hout]
  have hscope := F.c0.h20_0 member
  change parallel_generatedRoot_20 backend model params member.val ((inputs.1), (inputs.2.2.1), ((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)), ()) _ at hscope
  obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, constraints⟩ := hscope
  have hunit : (((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1))) = 1 := rfl
  have P_w_0_0 : ((inputs.1)) 0 0 = reducePoly modulus_0 8 (sm) := hmask
  have P_w_9_0 : w_9_0 0 0 = reducePoly modulus_0 8 (autoPlaintext (((MxxIR.selectInt ((Int.ofNat member.val)) [(7)]))) (sm)) := auto_plaintext_entry (by decide) (by decide) constraints.2.2.2.2.2.2.1 P_w_0_0
  have key0 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(7)]))]
      modulus_0 25 0 0 = w_3_0 := by
    rw [hashSample_eq constraints.2.2.1]
    rfl
  have key1 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(7)]))]
      modulus_0 25 1 0 = w_6_0 := by
    rw [hashSample_eq constraints.2.2.2.1]
    rfl
  have gadgetEq : w_10_0 = G :=
    gadget_matrix_unique constraints.2.2.2.2.2.2.2.1 hG
  unfold RootComponent
  rw [key0, key1, ← gadgetEq]
  apply component_encode hmask (by rfl)
    (by simpa only [one_mul, mul_one] using P_w_9_0)
    constraints.2.2.2.2.2.2.2.2.1 constraints.2.2.2.2.2.2.2.2.2.1
  simpa only [mul_one, one_mul] using constraints.2.2.2.2.2.2.2.2.2.2

theorem initial_component_helper_Aut_7_1 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 1) :
    RootComponent model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(7)]))] sm G
      (1) (outputs.2.2.2.2.2.2.2.2.2.2.2.2.1 member) 1 := by
  obtain ⟨witness, hbody⟩ := h
  let ctx : generatedRoot.Ctx backend model params := ⟨inputs, witness⟩
  have F := generatedRoot.facts ctx hbody
  rw [F.c2.hout]
  have hscope := F.c0.h21_0 member
  change parallel_generatedRoot_21 backend model params member.val ((inputs.1), (inputs.2.2.1), ((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)), ()) _ at hscope
  obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, constraints⟩ := hscope
  have hunit : (((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1))) = 1 := rfl
  have P_w_9_0 : ((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)) 0 0 = reducePoly modulus_0 8 (1) := one_entry
  have key0 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(7)]))]
      modulus_0 25 0 1 = w_3_0 := by
    rw [hashSample_eq constraints.2.2.1]
    rfl
  have key1 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(7)]))]
      modulus_0 25 1 1 = w_6_0 := by
    rw [hashSample_eq constraints.2.2.2.1]
    rfl
  have gadgetEq : w_10_0 = G :=
    gadget_matrix_unique constraints.2.2.2.2.1 hG
  unfold RootComponent
  rw [key0, key1, ← gadgetEq]
  apply component_encode hmask (by rfl)
    (by simpa only [one_mul, mul_one] using P_w_9_0)
    constraints.2.2.2.2.2.1 constraints.2.2.2.2.2.2.1
  simpa only [mul_one, one_mul] using constraints.2.2.2.2.2.2.2

/-- Every member of `helper_Aut(7)` is a root encoding with coefficient error at most one. -/
theorem initial_row_helper_Aut_7 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 1) :
    RootRow model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(7)]))] sm G
      (fun port ↦ fun _ _ ↦ reducePoly modulus_0 8
        ((1) * unitPair (autoPlaintext (((MxxIR.selectInt ((Int.ofNat (member.val))) [(7)]))) (sm)) port))
      ![(outputs.2.2.2.2.2.2.2.2.2.2.2.1 member), (outputs.2.2.2.2.2.2.2.2.2.2.2.2.1 member)] := by
  apply rootRow_pair_unit
  · simpa only [one_mul, mul_one] using initial_component_helper_Aut_7_0 h hmask hpay hG member
  · simpa only [one_mul, mul_one] using initial_component_helper_Aut_7_1 h hmask hpay hG member

theorem initial_component_helper_Aut_9_0 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 1) :
    RootComponent model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(9)]))] sm G
      ((1) * (autoPlaintext (((MxxIR.selectInt ((Int.ofNat (member.val))) [(9)]))) (sm))) (outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1 member) 0 := by
  obtain ⟨witness, hbody⟩ := h
  let ctx : generatedRoot.Ctx backend model params := ⟨inputs, witness⟩
  have F := generatedRoot.facts ctx hbody
  rw [F.c2.hout]
  have hscope := F.c0.h22_0 member
  change parallel_generatedRoot_22 backend model params member.val ((inputs.1), (inputs.2.2.1), ((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)), ()) _ at hscope
  obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, constraints⟩ := hscope
  have hunit : (((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1))) = 1 := rfl
  have P_w_0_0 : ((inputs.1)) 0 0 = reducePoly modulus_0 8 (sm) := hmask
  have P_w_9_0 : w_9_0 0 0 = reducePoly modulus_0 8 (autoPlaintext (((MxxIR.selectInt ((Int.ofNat member.val)) [(9)]))) (sm)) := auto_plaintext_entry (by decide) (by decide) constraints.2.2.2.2.2.2.1 P_w_0_0
  have key0 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(9)]))]
      modulus_0 25 0 0 = w_3_0 := by
    rw [hashSample_eq constraints.2.2.1]
    rfl
  have key1 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(9)]))]
      modulus_0 25 1 0 = w_6_0 := by
    rw [hashSample_eq constraints.2.2.2.1]
    rfl
  have gadgetEq : w_10_0 = G :=
    gadget_matrix_unique constraints.2.2.2.2.2.2.2.1 hG
  unfold RootComponent
  rw [key0, key1, ← gadgetEq]
  apply component_encode hmask (by rfl)
    (by simpa only [one_mul, mul_one] using P_w_9_0)
    constraints.2.2.2.2.2.2.2.2.1 constraints.2.2.2.2.2.2.2.2.2.1
  simpa only [mul_one, one_mul] using constraints.2.2.2.2.2.2.2.2.2.2

theorem initial_component_helper_Aut_9_1 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 1) :
    RootComponent model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(9)]))] sm G
      (1) (outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 member) 1 := by
  obtain ⟨witness, hbody⟩ := h
  let ctx : generatedRoot.Ctx backend model params := ⟨inputs, witness⟩
  have F := generatedRoot.facts ctx hbody
  rw [F.c2.hout]
  have hscope := F.c0.h23_0 member
  change parallel_generatedRoot_23 backend model params member.val ((inputs.1), (inputs.2.2.1), ((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)), ()) _ at hscope
  obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, constraints⟩ := hscope
  have hunit : (((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1))) = 1 := rfl
  have P_w_9_0 : ((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)) 0 0 = reducePoly modulus_0 8 (1) := one_entry
  have key0 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(9)]))]
      modulus_0 25 0 1 = w_3_0 := by
    rw [hashSample_eq constraints.2.2.1]
    rfl
  have key1 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(9)]))]
      modulus_0 25 1 1 = w_6_0 := by
    rw [hashSample_eq constraints.2.2.2.1]
    rfl
  have gadgetEq : w_10_0 = G :=
    gadget_matrix_unique constraints.2.2.2.2.1 hG
  unfold RootComponent
  rw [key0, key1, ← gadgetEq]
  apply component_encode hmask (by rfl)
    (by simpa only [one_mul, mul_one] using P_w_9_0)
    constraints.2.2.2.2.2.1 constraints.2.2.2.2.2.2.1
  simpa only [mul_one, one_mul] using constraints.2.2.2.2.2.2.2

/-- Every member of `helper_Aut(9)` is a root encoding with coefficient error at most one. -/
theorem initial_row_helper_Aut_9 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 1) :
    RootRow model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(9)]))] sm G
      (fun port ↦ fun _ _ ↦ reducePoly modulus_0 8
        ((1) * unitPair (autoPlaintext (((MxxIR.selectInt ((Int.ofNat (member.val))) [(9)]))) (sm)) port))
      ![(outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1 member), (outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 member)] := by
  apply rootRow_pair_unit
  · simpa only [one_mul, mul_one] using initial_component_helper_Aut_9_0 h hmask hpay hG member
  · simpa only [one_mul, mul_one] using initial_component_helper_Aut_9_1 h hmask hpay hG member

theorem initial_component_helper_ExpLayout_stride_log2_0_width_log2_3_0 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 8) :
    RootComponent model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(1), (3), (5), (7), (9), (11), (13), (15)]))] sm G
      ((1) * (autoPlaintext (((MxxIR.selectInt ((Int.ofNat (member.val))) [(1), (3), (5), (7), (9), (11), (13), (15)]))) (sm))) (outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 member) 0 := by
  obtain ⟨witness, hbody⟩ := h
  let ctx : generatedRoot.Ctx backend model params := ⟨inputs, witness⟩
  have F := generatedRoot.facts ctx hbody
  rw [F.c2.hout]
  have hscope := F.c1.h24_0 member
  change parallel_generatedRoot_24 backend model params member.val ((inputs.1), (inputs.2.2.1), ((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)), ()) _ at hscope
  obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, constraints⟩ := hscope
  have hunit : (((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1))) = 1 := rfl
  have P_w_0_0 : ((inputs.1)) 0 0 = reducePoly modulus_0 8 (sm) := hmask
  have P_w_9_0 : w_9_0 0 0 = reducePoly modulus_0 8 (autoPlaintext (((MxxIR.selectInt ((Int.ofNat member.val)) [(1), (3), (5), (7), (9), (11), (13), (15)]))) (sm)) := auto_plaintext_entry (by decide) (by decide) constraints.2.2.2.2.2.2.1 P_w_0_0
  have key0 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(1), (3), (5), (7), (9), (11), (13), (15)]))]
      modulus_0 25 0 0 = w_3_0 := by
    rw [hashSample_eq constraints.2.2.1]
    rfl
  have key1 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(1), (3), (5), (7), (9), (11), (13), (15)]))]
      modulus_0 25 1 0 = w_6_0 := by
    rw [hashSample_eq constraints.2.2.2.1]
    rfl
  have gadgetEq : w_10_0 = G :=
    gadget_matrix_unique constraints.2.2.2.2.2.2.2.1 hG
  unfold RootComponent
  rw [key0, key1, ← gadgetEq]
  apply component_encode hmask (by rfl)
    (by simpa only [one_mul, mul_one] using P_w_9_0)
    constraints.2.2.2.2.2.2.2.2.1 constraints.2.2.2.2.2.2.2.2.2.1
  simpa only [mul_one, one_mul] using constraints.2.2.2.2.2.2.2.2.2.2

theorem initial_component_helper_ExpLayout_stride_log2_0_width_log2_3_1 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 8) :
    RootComponent model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(1), (3), (5), (7), (9), (11), (13), (15)]))] sm G
      (1) (outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 member) 1 := by
  obtain ⟨witness, hbody⟩ := h
  let ctx : generatedRoot.Ctx backend model params := ⟨inputs, witness⟩
  have F := generatedRoot.facts ctx hbody
  rw [F.c2.hout]
  have hscope := F.c1.h25_0 member
  change parallel_generatedRoot_25 backend model params member.val ((inputs.1), (inputs.2.2.1), ((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)), ()) _ at hscope
  obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, constraints⟩ := hscope
  have hunit : (((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1))) = 1 := rfl
  have P_w_9_0 : ((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)) 0 0 = reducePoly modulus_0 8 (1) := one_entry
  have key0 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(1), (3), (5), (7), (9), (11), (13), (15)]))]
      modulus_0 25 0 1 = w_3_0 := by
    rw [hashSample_eq constraints.2.2.1]
    rfl
  have key1 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(1), (3), (5), (7), (9), (11), (13), (15)]))]
      modulus_0 25 1 1 = w_6_0 := by
    rw [hashSample_eq constraints.2.2.2.1]
    rfl
  have gadgetEq : w_10_0 = G :=
    gadget_matrix_unique constraints.2.2.2.2.1 hG
  unfold RootComponent
  rw [key0, key1, ← gadgetEq]
  apply component_encode hmask (by rfl)
    (by simpa only [one_mul, mul_one] using P_w_9_0)
    constraints.2.2.2.2.2.1 constraints.2.2.2.2.2.2.1
  simpa only [mul_one, one_mul] using constraints.2.2.2.2.2.2.2

/-- Every member of `helper_ExpLayout { stride_log2: 0, width_log2: 3 }` is a root encoding with coefficient error at most one. -/
theorem initial_row_helper_ExpLayout_stride_log2_0_width_log2_3 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 8) :
    RootRow model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(1), (3), (5), (7), (9), (11), (13), (15)]))] sm G
      (fun port ↦ fun _ _ ↦ reducePoly modulus_0 8
        ((1) * unitPair (autoPlaintext (((MxxIR.selectInt ((Int.ofNat (member.val))) [(1), (3), (5), (7), (9), (11), (13), (15)]))) (sm)) port))
      ![(outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 member), (outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 member)] := by
  apply rootRow_pair_unit
  · simpa only [one_mul, mul_one] using initial_component_helper_ExpLayout_stride_log2_0_width_log2_3_0 h hmask hpay hG member
  · simpa only [one_mul, mul_one] using initial_component_helper_ExpLayout_stride_log2_0_width_log2_3_1 h hmask hpay hG member

theorem initial_component_helper_ExpLayout_stride_log2_1_width_log2_2_0 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 4) :
    RootComponent model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(1), (3), (5), (15)]))] sm G
      ((1) * (autoPlaintext (((MxxIR.selectInt ((Int.ofNat (member.val))) [(1), (3), (5), (15)]))) (sm))) (outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 member) 0 := by
  obtain ⟨witness, hbody⟩ := h
  let ctx : generatedRoot.Ctx backend model params := ⟨inputs, witness⟩
  have F := generatedRoot.facts ctx hbody
  rw [F.c2.hout]
  have hscope := F.c1.h26_0 member
  change parallel_generatedRoot_26 backend model params member.val ((inputs.1), (inputs.2.2.1), ((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)), ()) _ at hscope
  obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, constraints⟩ := hscope
  have hunit : (((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1))) = 1 := rfl
  have P_w_0_0 : ((inputs.1)) 0 0 = reducePoly modulus_0 8 (sm) := hmask
  have P_w_9_0 : w_9_0 0 0 = reducePoly modulus_0 8 (autoPlaintext (((MxxIR.selectInt ((Int.ofNat member.val)) [(1), (3), (5), (15)]))) (sm)) := auto_plaintext_entry (by decide) (by decide) constraints.2.2.2.2.2.2.1 P_w_0_0
  have key0 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(1), (3), (5), (15)]))]
      modulus_0 25 0 0 = w_3_0 := by
    rw [hashSample_eq constraints.2.2.1]
    rfl
  have key1 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(1), (3), (5), (15)]))]
      modulus_0 25 1 0 = w_6_0 := by
    rw [hashSample_eq constraints.2.2.2.1]
    rfl
  have gadgetEq : w_10_0 = G :=
    gadget_matrix_unique constraints.2.2.2.2.2.2.2.1 hG
  unfold RootComponent
  rw [key0, key1, ← gadgetEq]
  apply component_encode hmask (by rfl)
    (by simpa only [one_mul, mul_one] using P_w_9_0)
    constraints.2.2.2.2.2.2.2.2.1 constraints.2.2.2.2.2.2.2.2.2.1
  simpa only [mul_one, one_mul] using constraints.2.2.2.2.2.2.2.2.2.2

theorem initial_component_helper_ExpLayout_stride_log2_1_width_log2_2_1 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 4) :
    RootComponent model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(1), (3), (5), (15)]))] sm G
      (1) (outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 member) 1 := by
  obtain ⟨witness, hbody⟩ := h
  let ctx : generatedRoot.Ctx backend model params := ⟨inputs, witness⟩
  have F := generatedRoot.facts ctx hbody
  rw [F.c2.hout]
  have hscope := F.c1.h27_0 member
  change parallel_generatedRoot_27 backend model params member.val ((inputs.1), (inputs.2.2.1), ((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)), ()) _ at hscope
  obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, constraints⟩ := hscope
  have hunit : (((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1))) = 1 := rfl
  have P_w_9_0 : ((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)) 0 0 = reducePoly modulus_0 8 (1) := one_entry
  have key0 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(1), (3), (5), (15)]))]
      modulus_0 25 0 1 = w_3_0 := by
    rw [hashSample_eq constraints.2.2.1]
    rfl
  have key1 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(1), (3), (5), (15)]))]
      modulus_0 25 1 1 = w_6_0 := by
    rw [hashSample_eq constraints.2.2.2.1]
    rfl
  have gadgetEq : w_10_0 = G :=
    gadget_matrix_unique constraints.2.2.2.2.1 hG
  unfold RootComponent
  rw [key0, key1, ← gadgetEq]
  apply component_encode hmask (by rfl)
    (by simpa only [one_mul, mul_one] using P_w_9_0)
    constraints.2.2.2.2.2.1 constraints.2.2.2.2.2.2.1
  simpa only [mul_one, one_mul] using constraints.2.2.2.2.2.2.2

/-- Every member of `helper_ExpLayout { stride_log2: 1, width_log2: 2 }` is a root encoding with coefficient error at most one. -/
theorem initial_row_helper_ExpLayout_stride_log2_1_width_log2_2 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 4) :
    RootRow model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97] [.integer ((MxxIR.selectInt ((Int.ofNat (member.val))) [(1), (3), (5), (15)]))] sm G
      (fun port ↦ fun _ _ ↦ reducePoly modulus_0 8
        ((1) * unitPair (autoPlaintext (((MxxIR.selectInt ((Int.ofNat (member.val))) [(1), (3), (5), (15)]))) (sm)) port))
      ![(outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 member), (outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 member)] := by
  apply rootRow_pair_unit
  · simpa only [one_mul, mul_one] using initial_component_helper_ExpLayout_stride_log2_1_width_log2_2_0 h hmask hpay hG member
  · simpa only [one_mul, mul_one] using initial_component_helper_ExpLayout_stride_log2_1_width_log2_2_1 h hmask hpay hG member

theorem initial_component_row_dio2_label_0_0 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 2) :
    RootComponent model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48] [.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat (member.val)))] sm G
      ((message 1 (MxxIR.selectInt (if (decide (((Int.ofNat (member.val))) = (1))) then 1 else 0) [((intFamilyValue (((inputs.2.2.2.1))) ((((((Int.ofNat (member.val))) % (1))) * (5)))) * (1)), ((0) + (0))])) * (sm)) (outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 member) 0 := by
  obtain ⟨witness, hbody⟩ := h
  let ctx : generatedRoot.Ctx backend model params := ⟨inputs, witness⟩
  have F := generatedRoot.facts ctx hbody
  rw [F.c2.hout]
  have hscope := F.c1.h29_0 member
  change parallel_generatedRoot_29 backend model params member.val ((inputs.1), (inputs.2.2.1), ((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)), (inputs.2.2.2.1), ()) _ at hscope
  obtain ⟨w_2_0, w_5_0, w_18_0, w_24_0, w_27_0, w_30_0, w_31_0, constraints⟩ := hscope
  have hunit : (((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1))) = 1 := rfl
  have P_w_8_0 : ((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)) 0 0 = reducePoly modulus_0 8 (1) := one_entry
  have I_w_9_0 : ((Int.ofNat member.val)) = (Int.ofNat member.val) := rfl
  have I_w_10_0 : (1) = 1 := rfl
  have I_w_11_0 : (decide (((Int.ofNat member.val)) = (1))) = decide (((Int.ofNat member.val)) = (1)) := rfl
  have I_w_12_0 : (if (decide (((Int.ofNat member.val)) = (1))) then 1 else 0) = if (decide (((Int.ofNat member.val)) = (1))) then 1 else 0 := rfl
  have I_w_14_0 : (1) = 1 := rfl
  have I_w_15_0 : ((((Int.ofNat member.val)) % (1))) = (((Int.ofNat member.val)) % (1)) := rfl
  have I_w_16_0 : (5) = 5 := rfl
  have I_w_17_0 : ((((((Int.ofNat member.val)) % (1))) * (5))) = (((((Int.ofNat member.val)) % (1))) * (5)) := rfl
  have I_w_18_0 : w_18_0 = intFamilyValue (((inputs.2.2.2.1))) ((((((Int.ofNat member.val)) % (1))) * (5))) := (int_family_value constraints.2.2.2.2.2.1).trans (by simp only [])
  have I_w_19_0 : (1) = 1 := by change (1) = _; simp only [I_w_18_0]
  have I_w_20_0 : ((w_18_0 * (1))) = ((intFamilyValue (((inputs.2.2.2.1))) ((((((Int.ofNat member.val)) % (1))) * (5)))) * (1)) := by change ((w_18_0 * (1))) = _; simp only [I_w_18_0]
  have I_w_21_0 : (0) = 0 := by change (0) = _; simp only [I_w_18_0]
  have I_w_22_0 : (0) = 0 := by change (0) = _; simp only [I_w_18_0]
  have I_w_23_0 : (((0) + (0))) = ((0) + (0)) := by change (((0) + (0))) = _; simp only [I_w_18_0]
  have I_w_24_0 : w_24_0 = MxxIR.selectInt (if (decide (((Int.ofNat member.val)) = (1))) then 1 else 0) [((intFamilyValue (((inputs.2.2.2.1))) ((((((Int.ofNat member.val)) % (1))) * (5)))) * (1)), ((0) + (0))] := (select_value constraints.2.2.2.2.2.2.2.2.2.1).trans (by simp only [I_w_18_0])
  have P_w_25_0 : (MxxRuntime.multiplyMonomial ((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)) w_24_0) 0 0 = reducePoly modulus_0 8 ((1 * message 1 (MxxIR.selectInt (if (decide (((Int.ofNat member.val)) = (1))) then 1 else 0) [((intFamilyValue (((inputs.2.2.2.1))) ((((((Int.ofNat member.val)) % (1))) * (5)))) * (1)), ((0) + (0))]))) := (mono_entry P_w_8_0 _).trans (by rw [I_w_24_0])
  have P_w_0_0 : ((inputs.1)) 0 0 = reducePoly modulus_0 8 (sm) := hmask
  have P_w_26_0 : (MxxRuntime.matrixMulScalarLeft (MxxRuntime.multiplyMonomial ((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)) w_24_0) ((inputs.1))) 0 0 = reducePoly modulus_0 8 (((1 * message 1 (MxxIR.selectInt (if (decide (((Int.ofNat member.val)) = (1))) then 1 else 0) [((intFamilyValue (((inputs.2.2.2.1))) ((((((Int.ofNat member.val)) % (1))) * (5)))) * (1)), ((0) + (0))])) * sm)) := msl_entry P_w_25_0 P_w_0_0
  have key0 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48] [.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat (member.val)))]
      modulus_0 25 0 0 = w_2_0 := by
    rw [hashSample_eq constraints.1]
    rfl
  have key1 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48] [.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat (member.val)))]
      modulus_0 25 1 0 = w_5_0 := by
    rw [hashSample_eq constraints.2.1]
    rfl
  have gadgetEq : w_27_0 = G :=
    gadget_matrix_unique constraints.2.2.2.2.2.2.2.2.2.2.1 hG
  unfold RootComponent
  rw [key0, key1, ← gadgetEq]
  apply component_encode hmask (by rfl)
    (by simpa only [one_mul, mul_one] using P_w_26_0)
    constraints.2.2.2.2.2.2.2.2.2.2.2.1 constraints.2.2.2.2.2.2.2.2.2.2.2.2.1
  simpa only [mul_one, one_mul] using constraints.2.2.2.2.2.2.2.2.2.2.2.2.2

theorem initial_component_row_dio2_label_0_1 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 2) :
    RootComponent model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48] [.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat (member.val)))] sm G
      (message 1 (MxxIR.selectInt (if (decide (((Int.ofNat (member.val))) = (1))) then 1 else 0) [((intFamilyValue (((inputs.2.2.2.1))) ((((((Int.ofNat (member.val))) % (1))) * (5)))) * (1)), ((0) + (0))])) (outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 member) 1 := by
  obtain ⟨witness, hbody⟩ := h
  let ctx : generatedRoot.Ctx backend model params := ⟨inputs, witness⟩
  have F := generatedRoot.facts ctx hbody
  rw [F.c2.hout]
  have hscope := F.c1.h30_0 member
  change parallel_generatedRoot_30 backend model params member.val ((inputs.1), (inputs.2.2.1), ((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)), (inputs.2.2.2.1), ()) _ at hscope
  obtain ⟨w_2_0, w_5_0, w_18_0, w_24_0, w_27_0, w_30_0, w_31_0, constraints⟩ := hscope
  have hunit : (((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1))) = 1 := rfl
  have P_w_8_0 : ((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)) 0 0 = reducePoly modulus_0 8 (1) := one_entry
  have I_w_9_0 : ((Int.ofNat member.val)) = (Int.ofNat member.val) := rfl
  have I_w_10_0 : (1) = 1 := rfl
  have I_w_11_0 : (decide (((Int.ofNat member.val)) = (1))) = decide (((Int.ofNat member.val)) = (1)) := rfl
  have I_w_12_0 : (if (decide (((Int.ofNat member.val)) = (1))) then 1 else 0) = if (decide (((Int.ofNat member.val)) = (1))) then 1 else 0 := rfl
  have I_w_14_0 : (1) = 1 := rfl
  have I_w_15_0 : ((((Int.ofNat member.val)) % (1))) = (((Int.ofNat member.val)) % (1)) := rfl
  have I_w_16_0 : (5) = 5 := rfl
  have I_w_17_0 : ((((((Int.ofNat member.val)) % (1))) * (5))) = (((((Int.ofNat member.val)) % (1))) * (5)) := rfl
  have I_w_18_0 : w_18_0 = intFamilyValue (((inputs.2.2.2.1))) ((((((Int.ofNat member.val)) % (1))) * (5))) := (int_family_value constraints.2.2.2.2.2.1).trans (by simp only [])
  have I_w_19_0 : (1) = 1 := by change (1) = _; simp only [I_w_18_0]
  have I_w_20_0 : ((w_18_0 * (1))) = ((intFamilyValue (((inputs.2.2.2.1))) ((((((Int.ofNat member.val)) % (1))) * (5)))) * (1)) := by change ((w_18_0 * (1))) = _; simp only [I_w_18_0]
  have I_w_21_0 : (0) = 0 := by change (0) = _; simp only [I_w_18_0]
  have I_w_22_0 : (0) = 0 := by change (0) = _; simp only [I_w_18_0]
  have I_w_23_0 : (((0) + (0))) = ((0) + (0)) := by change (((0) + (0))) = _; simp only [I_w_18_0]
  have I_w_24_0 : w_24_0 = MxxIR.selectInt (if (decide (((Int.ofNat member.val)) = (1))) then 1 else 0) [((intFamilyValue (((inputs.2.2.2.1))) ((((((Int.ofNat member.val)) % (1))) * (5)))) * (1)), ((0) + (0))] := (select_value constraints.2.2.2.2.2.2.2.2.2.1).trans (by simp only [I_w_18_0])
  have P_w_25_0 : (MxxRuntime.multiplyMonomial ((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)) w_24_0) 0 0 = reducePoly modulus_0 8 ((1 * message 1 (MxxIR.selectInt (if (decide (((Int.ofNat member.val)) = (1))) then 1 else 0) [((intFamilyValue (((inputs.2.2.2.1))) ((((((Int.ofNat member.val)) % (1))) * (5)))) * (1)), ((0) + (0))]))) := (mono_entry P_w_8_0 _).trans (by rw [I_w_24_0])
  have P_w_4_0 : (((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1))) 0 0 = reducePoly modulus_0 8 (1) := by rw [hunit]; exact one_entry
  have P_w_26_0 : (MxxRuntime.matrixMulScalarLeft (MxxRuntime.multiplyMonomial ((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)) w_24_0) (((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)))) 0 0 = reducePoly modulus_0 8 (((1 * message 1 (MxxIR.selectInt (if (decide (((Int.ofNat member.val)) = (1))) then 1 else 0) [((intFamilyValue (((inputs.2.2.2.1))) ((((((Int.ofNat member.val)) % (1))) * (5)))) * (1)), ((0) + (0))])) * 1)) := msl_entry P_w_25_0 P_w_4_0
  have key0 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48] [.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat (member.val)))]
      modulus_0 25 0 1 = w_2_0 := by
    rw [hashSample_eq constraints.1]
    rfl
  have key1 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48] [.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat (member.val)))]
      modulus_0 25 1 1 = w_5_0 := by
    rw [hashSample_eq constraints.2.1]
    rfl
  have gadgetEq : w_27_0 = G :=
    gadget_matrix_unique constraints.2.2.2.2.2.2.2.2.2.2.1 hG
  unfold RootComponent
  rw [key0, key1, ← gadgetEq]
  apply component_encode hmask (by rfl)
    (by simpa only [one_mul, mul_one] using P_w_26_0)
    constraints.2.2.2.2.2.2.2.2.2.2.2.1 constraints.2.2.2.2.2.2.2.2.2.2.2.2.1
  simpa only [mul_one, one_mul] using constraints.2.2.2.2.2.2.2.2.2.2.2.2.2

/-- Every member of `row_dio2/label-0` is a root encoding with coefficient error at most one. -/
theorem initial_row_row_dio2_label_0 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 2) :
    RootRow model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48] [.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat (member.val)))] sm G
      (fun port ↦ fun _ _ ↦ reducePoly modulus_0 8
        ((message 1 (MxxIR.selectInt (if (decide (((Int.ofNat (member.val))) = (1))) then 1 else 0) [((intFamilyValue (((inputs.2.2.2.1))) ((((((Int.ofNat (member.val))) % (1))) * (5)))) * (1)), ((0) + (0))])) * unitPair (sm) port))
      ![(outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 member), (outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 member)] := by
  apply rootRow_pair_unit
  · simpa only [one_mul, mul_one] using initial_component_row_dio2_label_0_0 h hmask hpay hG member
  · simpa only [one_mul, mul_one] using initial_component_row_dio2_label_0_1 h hmask hpay hG member

theorem initial_component_row_dio2_label_0 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 4) :
    RootComponent model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108] [.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat (member.val)))] sm G
      ((message 1 (((intFamilyValue (((inputs.2.2.2.1))) ((((((((((Int.ofNat (member.val))) / (4))) * (5))) + ((((Int.ofNat (member.val))) % (4))))) + (1)))) * (1)))) * (sm)) (outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 member) 0 := by
  obtain ⟨witness, hbody⟩ := h
  let ctx : generatedRoot.Ctx backend model params := ⟨inputs, witness⟩
  have F := generatedRoot.facts ctx hbody
  rw [F.c2.hout]
  have hscope := F.c1.h31_0 member
  change parallel_generatedRoot_31 backend model params member.val ((inputs.1), (inputs.2.2.1), ((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)), (inputs.2.2.2.1), ()) _ at hscope
  obtain ⟨w_2_0, w_5_0, w_20_0, w_25_0, w_28_0, w_29_0, constraints⟩ := hscope
  have hunit : (((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1))) = 1 := rfl
  have P_w_8_0 : ((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)) 0 0 = reducePoly modulus_0 8 (1) := one_entry
  have I_w_10_0 : ((Int.ofNat member.val)) = (Int.ofNat member.val) := rfl
  have I_w_11_0 : (4) = 4 := rfl
  have I_w_12_0 : ((((Int.ofNat member.val)) / (4))) = (((Int.ofNat member.val)) / (4)) := rfl
  have I_w_13_0 : (5) = 5 := rfl
  have I_w_14_0 : ((((((Int.ofNat member.val)) / (4))) * (5))) = (((((Int.ofNat member.val)) / (4))) * (5)) := rfl
  have I_w_15_0 : (4) = 4 := rfl
  have I_w_16_0 : ((((Int.ofNat member.val)) % (4))) = (((Int.ofNat member.val)) % (4)) := rfl
  have I_w_17_0 : ((((((((Int.ofNat member.val)) / (4))) * (5))) + ((((Int.ofNat member.val)) % (4))))) = (((((((Int.ofNat member.val)) / (4))) * (5))) + ((((Int.ofNat member.val)) % (4)))) := rfl
  have I_w_18_0 : (1) = 1 := rfl
  have I_w_19_0 : ((((((((((Int.ofNat member.val)) / (4))) * (5))) + ((((Int.ofNat member.val)) % (4))))) + (1))) = (((((((((Int.ofNat member.val)) / (4))) * (5))) + ((((Int.ofNat member.val)) % (4))))) + (1)) := rfl
  have I_w_20_0 : w_20_0 = intFamilyValue (((inputs.2.2.2.1))) ((((((((((Int.ofNat member.val)) / (4))) * (5))) + ((((Int.ofNat member.val)) % (4))))) + (1))) := (int_family_value constraints.2.2.2.2.2.2.1).trans (by simp only [])
  have I_w_21_0 : (1) = 1 := by change (1) = _; simp only [I_w_20_0]
  have I_w_22_0 : ((w_20_0 * (1))) = ((intFamilyValue (((inputs.2.2.2.1))) ((((((((((Int.ofNat member.val)) / (4))) * (5))) + ((((Int.ofNat member.val)) % (4))))) + (1)))) * (1)) := by change ((w_20_0 * (1))) = _; simp only [I_w_20_0]
  have P_w_23_0 : (MxxRuntime.multiplyMonomial ((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)) ((w_20_0 * (1)))) 0 0 = reducePoly modulus_0 8 ((1 * message 1 (((intFamilyValue (((inputs.2.2.2.1))) ((((((((((Int.ofNat member.val)) / (4))) * (5))) + ((((Int.ofNat member.val)) % (4))))) + (1)))) * (1))))) := (mono_entry P_w_8_0 _).trans (by rw [I_w_22_0])
  have P_w_0_0 : ((inputs.1)) 0 0 = reducePoly modulus_0 8 (sm) := hmask
  have P_w_24_0 : (MxxRuntime.matrixMulScalarLeft (MxxRuntime.multiplyMonomial ((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)) ((w_20_0 * (1)))) ((inputs.1))) 0 0 = reducePoly modulus_0 8 (((1 * message 1 (((intFamilyValue (((inputs.2.2.2.1))) ((((((((((Int.ofNat member.val)) / (4))) * (5))) + ((((Int.ofNat member.val)) % (4))))) + (1)))) * (1)))) * sm)) := msl_entry P_w_23_0 P_w_0_0
  have key0 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108] [.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat (member.val)))]
      modulus_0 25 0 0 = w_2_0 := by
    rw [hashSample_eq constraints.1]
    rfl
  have key1 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108] [.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat (member.val)))]
      modulus_0 25 1 0 = w_5_0 := by
    rw [hashSample_eq constraints.2.1]
    rfl
  have gadgetEq : w_25_0 = G :=
    gadget_matrix_unique constraints.2.2.2.2.2.2.2.1 hG
  unfold RootComponent
  rw [key0, key1, ← gadgetEq]
  apply component_encode hmask (by rfl)
    (by simpa only [one_mul, mul_one] using P_w_24_0)
    constraints.2.2.2.2.2.2.2.2.1 constraints.2.2.2.2.2.2.2.2.2.1
  simpa only [mul_one, one_mul] using constraints.2.2.2.2.2.2.2.2.2.2

theorem initial_component_row_dio2_label_1 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 4) :
    RootComponent model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108] [.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat (member.val)))] sm G
      (message 1 (((intFamilyValue (((inputs.2.2.2.1))) ((((((((((Int.ofNat (member.val))) / (4))) * (5))) + ((((Int.ofNat (member.val))) % (4))))) + (1)))) * (1)))) (outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 member) 1 := by
  obtain ⟨witness, hbody⟩ := h
  let ctx : generatedRoot.Ctx backend model params := ⟨inputs, witness⟩
  have F := generatedRoot.facts ctx hbody
  rw [F.c2.hout]
  have hscope := F.c1.h32_0 member
  change parallel_generatedRoot_32 backend model params member.val ((inputs.1), (inputs.2.2.1), ((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)), (inputs.2.2.2.1), ()) _ at hscope
  obtain ⟨w_2_0, w_5_0, w_20_0, w_25_0, w_28_0, w_29_0, constraints⟩ := hscope
  have hunit : (((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1))) = 1 := rfl
  have P_w_8_0 : ((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)) 0 0 = reducePoly modulus_0 8 (1) := one_entry
  have I_w_10_0 : ((Int.ofNat member.val)) = (Int.ofNat member.val) := rfl
  have I_w_11_0 : (4) = 4 := rfl
  have I_w_12_0 : ((((Int.ofNat member.val)) / (4))) = (((Int.ofNat member.val)) / (4)) := rfl
  have I_w_13_0 : (5) = 5 := rfl
  have I_w_14_0 : ((((((Int.ofNat member.val)) / (4))) * (5))) = (((((Int.ofNat member.val)) / (4))) * (5)) := rfl
  have I_w_15_0 : (4) = 4 := rfl
  have I_w_16_0 : ((((Int.ofNat member.val)) % (4))) = (((Int.ofNat member.val)) % (4)) := rfl
  have I_w_17_0 : ((((((((Int.ofNat member.val)) / (4))) * (5))) + ((((Int.ofNat member.val)) % (4))))) = (((((((Int.ofNat member.val)) / (4))) * (5))) + ((((Int.ofNat member.val)) % (4)))) := rfl
  have I_w_18_0 : (1) = 1 := rfl
  have I_w_19_0 : ((((((((((Int.ofNat member.val)) / (4))) * (5))) + ((((Int.ofNat member.val)) % (4))))) + (1))) = (((((((((Int.ofNat member.val)) / (4))) * (5))) + ((((Int.ofNat member.val)) % (4))))) + (1)) := rfl
  have I_w_20_0 : w_20_0 = intFamilyValue (((inputs.2.2.2.1))) ((((((((((Int.ofNat member.val)) / (4))) * (5))) + ((((Int.ofNat member.val)) % (4))))) + (1))) := (int_family_value constraints.2.2.2.2.2.2.1).trans (by simp only [])
  have I_w_21_0 : (1) = 1 := by change (1) = _; simp only [I_w_20_0]
  have I_w_22_0 : ((w_20_0 * (1))) = ((intFamilyValue (((inputs.2.2.2.1))) ((((((((((Int.ofNat member.val)) / (4))) * (5))) + ((((Int.ofNat member.val)) % (4))))) + (1)))) * (1)) := by change ((w_20_0 * (1))) = _; simp only [I_w_20_0]
  have P_w_23_0 : (MxxRuntime.multiplyMonomial ((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)) ((w_20_0 * (1)))) 0 0 = reducePoly modulus_0 8 ((1 * message 1 (((intFamilyValue (((inputs.2.2.2.1))) ((((((((((Int.ofNat member.val)) / (4))) * (5))) + ((((Int.ofNat member.val)) % (4))))) + (1)))) * (1))))) := (mono_entry P_w_8_0 _).trans (by rw [I_w_22_0])
  have P_w_4_0 : (((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1))) 0 0 = reducePoly modulus_0 8 (1) := by rw [hunit]; exact one_entry
  have P_w_24_0 : (MxxRuntime.matrixMulScalarLeft (MxxRuntime.multiplyMonomial ((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)) ((w_20_0 * (1)))) (((1 : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)))) 0 0 = reducePoly modulus_0 8 (((1 * message 1 (((intFamilyValue (((inputs.2.2.2.1))) ((((((((((Int.ofNat member.val)) / (4))) * (5))) + ((((Int.ofNat member.val)) % (4))))) + (1)))) * (1)))) * 1)) := msl_entry P_w_23_0 P_w_4_0
  have key0 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108] [.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat (member.val)))]
      modulus_0 25 0 1 = w_2_0 := by
    rw [hashSample_eq constraints.1]
    rfl
  have key1 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108] [.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat (member.val)))]
      modulus_0 25 1 1 = w_5_0 := by
    rw [hashSample_eq constraints.2.1]
    rfl
  have gadgetEq : w_25_0 = G :=
    gadget_matrix_unique constraints.2.2.2.2.2.2.2.1 hG
  unfold RootComponent
  rw [key0, key1, ← gadgetEq]
  apply component_encode hmask (by rfl)
    (by simpa only [one_mul, mul_one] using P_w_24_0)
    constraints.2.2.2.2.2.2.2.2.1 constraints.2.2.2.2.2.2.2.2.2.1
  simpa only [mul_one, one_mul] using constraints.2.2.2.2.2.2.2.2.2.2

/-- Every member of `row_dio2/label` is a root encoding with coefficient error at most one. -/
theorem initial_row_row_dio2_label {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) (member : Fin 4) :
    RootRow model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108] [.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat (member.val)))] sm G
      (fun port ↦ fun _ _ ↦ reducePoly modulus_0 8
        ((message 1 (((intFamilyValue (((inputs.2.2.2.1))) ((((((((((Int.ofNat (member.val))) / (4))) * (5))) + ((((Int.ofNat (member.val))) % (4))))) + (1)))) * (1)))) * unitPair (sm) port))
      ![(outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 member), (outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 member)] := by
  apply rootRow_pair_unit
  · simpa only [one_mul, mul_one] using initial_component_row_dio2_label_0 h hmask hpay hG member
  · simpa only [one_mul, mul_one] using initial_component_row_dio2_label_1 h hmask hpay hG member

theorem initial_component_row_dio2_one_private_0 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) :
    RootComponent model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101] [] sm G
      ((1) * (tp)) outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 0 := by
  obtain ⟨witness, hbody⟩ := h
  let ctx : generatedRoot.Ctx backend model params := ⟨inputs, witness⟩
  have F := generatedRoot.facts ctx hbody
  rw [F.c2.hout]
  have P_w_38_0 : (generatedRoot.w_38_0 ctx) 0 0 = reducePoly modulus_0 8 (1) := one_entry
  have P_w_39_0 : (generatedRoot.w_39_0 ctx) 0 0 = reducePoly modulus_0 8 (tp) := hpay
  have P_w_40_0 : (generatedRoot.w_40_0 ctx) 0 0 = reducePoly modulus_0 8 ((1 * tp)) := msl_entry P_w_38_0 P_w_39_0
  have key0 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101] []
      modulus_0 25 0 0 = (generatedRoot.w_33_0 ctx) := by
    rw [hashSample_eq F.c1.h33_0]
    rfl
  have key1 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101] []
      modulus_0 25 1 0 = (generatedRoot.w_35_0 ctx) := by
    rw [hashSample_eq F.c1.h35_0]
    rfl
  have gadgetEq : (generatedRoot.w_41_0 ctx) = G :=
    gadget_matrix_unique F.c1.h41_0 hG
  unfold RootComponent
  rw [key0, key1, ← gadgetEq]
  apply component_encode hmask (by rfl)
    (by simpa only [one_mul, mul_one] using P_w_40_0)
    F.c1.h44_0 F.c1.h45_0
  rfl

theorem initial_component_row_dio2_one_private_1 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) :
    RootComponent model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101] [] sm G
      (1) outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 1 := by
  obtain ⟨witness, hbody⟩ := h
  let ctx : generatedRoot.Ctx backend model params := ⟨inputs, witness⟩
  have F := generatedRoot.facts ctx hbody
  rw [F.c2.hout]
  have P_w_53_0 : (generatedRoot.w_53_0 ctx) 0 0 = reducePoly modulus_0 8 (1) := one_entry
  have P_w_54_0 : (generatedRoot.w_54_0 ctx) 0 0 = reducePoly modulus_0 8 (1) := one_entry
  have P_w_55_0 : (generatedRoot.w_55_0 ctx) 0 0 = reducePoly modulus_0 8 ((1 * 1)) := msl_entry P_w_53_0 P_w_54_0
  have key0 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101] []
      modulus_0 25 0 1 = (generatedRoot.w_48_0 ctx) := by
    rw [hashSample_eq F.c1.h48_0]
    rfl
  have key1 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101] []
      modulus_0 25 1 1 = (generatedRoot.w_50_0 ctx) := by
    rw [hashSample_eq F.c1.h50_0]
    rfl
  have gadgetEq : (generatedRoot.w_56_0 ctx) = G :=
    gadget_matrix_unique F.c1.h56_0 hG
  unfold RootComponent
  rw [key0, key1, ← gadgetEq]
  apply component_encode hmask (by rfl)
    (by simpa only [one_mul, mul_one] using P_w_55_0)
    F.c1.h59_0 F.c1.h60_0
  rfl

/-- Every member of `row_dio2/one-private` is a root encoding with coefficient error at most one. -/
theorem initial_row_row_dio2_one_private {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) :
    RootRow model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101] [] sm G
      (fun port ↦ fun _ _ ↦ reducePoly modulus_0 8
        ((1) * unitPair (tp) port))
      ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1] := by
  apply rootRow_pair_unit
  · simpa only [one_mul, mul_one] using initial_component_row_dio2_one_private_0 h hmask hpay hG
  · simpa only [one_mul, mul_one] using initial_component_row_dio2_one_private_1 h hmask hpay hG

theorem initial_component_row_dio2_state_0 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) :
    RootComponent model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101] [] sm G
      ((message 1 (((inputs.2.2.2.2.2.1) * (2)))) * (tp)) outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 0 := by
  obtain ⟨witness, hbody⟩ := h
  let ctx : generatedRoot.Ctx backend model params := ⟨inputs, witness⟩
  have F := generatedRoot.facts ctx hbody
  rw [F.c2.hout]
  have P_w_68_0 : (generatedRoot.w_68_0 ctx) 0 0 = reducePoly modulus_0 8 (1) := one_entry
  have I_w_69_0 : (generatedRoot.w_69_0 ctx) = inputs.2.2.2.2.2.1 := rfl
  have I_w_70_0 : (generatedRoot.w_70_0 ctx) = 2 := rfl
  have I_w_71_0 : (generatedRoot.w_71_0 ctx) = ((inputs.2.2.2.2.2.1) * (2)) := rfl
  have P_w_72_0 : (generatedRoot.w_72_0 ctx) 0 0 = reducePoly modulus_0 8 ((1 * message 1 (((inputs.2.2.2.2.2.1) * (2))))) := (mono_entry P_w_68_0 _).trans (by rw [I_w_71_0])
  have P_w_39_0 : (generatedRoot.w_39_0 ctx) 0 0 = reducePoly modulus_0 8 (tp) := hpay
  have P_w_73_0 : (generatedRoot.w_73_0 ctx) 0 0 = reducePoly modulus_0 8 (((1 * message 1 (((inputs.2.2.2.2.2.1) * (2)))) * tp)) := msl_entry P_w_72_0 P_w_39_0
  have key0 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101] []
      modulus_0 25 0 0 = (generatedRoot.w_63_0 ctx) := by
    rw [hashSample_eq F.c1.h63_0]
    rfl
  have key1 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101] []
      modulus_0 25 1 0 = (generatedRoot.w_65_0 ctx) := by
    rw [hashSample_eq F.c1.h65_0]
    rfl
  have gadgetEq : (generatedRoot.w_74_0 ctx) = G :=
    gadget_matrix_unique F.c1.h74_0 hG
  unfold RootComponent
  rw [key0, key1, ← gadgetEq]
  apply component_encode hmask (by rfl)
    (by simpa only [one_mul, mul_one] using P_w_73_0)
    F.c1.h77_0 F.c1.h78_0
  rfl

theorem initial_component_row_dio2_state_1 {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) :
    RootComponent model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101] [] sm G
      (message 1 (((inputs.2.2.2.2.2.1) * (2)))) outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 1 := by
  obtain ⟨witness, hbody⟩ := h
  let ctx : generatedRoot.Ctx backend model params := ⟨inputs, witness⟩
  have F := generatedRoot.facts ctx hbody
  rw [F.c2.hout]
  have P_w_86_0 : (generatedRoot.w_86_0 ctx) 0 0 = reducePoly modulus_0 8 (1) := one_entry
  have I_w_69_0 : (generatedRoot.w_69_0 ctx) = inputs.2.2.2.2.2.1 := rfl
  have I_w_87_0 : (generatedRoot.w_87_0 ctx) = 2 := rfl
  have I_w_88_0 : (generatedRoot.w_88_0 ctx) = ((inputs.2.2.2.2.2.1) * (2)) := rfl
  have P_w_89_0 : (generatedRoot.w_89_0 ctx) 0 0 = reducePoly modulus_0 8 ((1 * message 1 (((inputs.2.2.2.2.2.1) * (2))))) := (mono_entry P_w_86_0 _).trans (by rw [I_w_88_0])
  have P_w_54_0 : (generatedRoot.w_54_0 ctx) 0 0 = reducePoly modulus_0 8 (1) := one_entry
  have P_w_90_0 : (generatedRoot.w_90_0 ctx) 0 0 = reducePoly modulus_0 8 (((1 * message 1 (((inputs.2.2.2.2.2.1) * (2)))) * 1)) := msl_entry P_w_89_0 P_w_54_0
  have key0 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101] []
      modulus_0 25 0 1 = (generatedRoot.w_81_0 ctx) := by
    rw [hashSample_eq F.c1.h81_0]
    rfl
  have key1 : rootKey model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101] []
      modulus_0 25 1 1 = (generatedRoot.w_83_0 ctx) := by
    rw [hashSample_eq F.c1.h83_0]
    rfl
  have gadgetEq : (generatedRoot.w_91_0 ctx) = G :=
    gadget_matrix_unique F.c2.h91_0 hG
  unfold RootComponent
  rw [key0, key1, ← gadgetEq]
  apply component_encode hmask (by rfl)
    (by simpa only [one_mul, mul_one] using P_w_90_0)
    F.c2.h94_0 F.c2.h95_0
  rfl

/-- Every member of `row_dio2/state` is a root encoding with coefficient error at most one. -/
theorem initial_row_row_dio2_state {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm tp : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm)
    (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)
    {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 G) :
    RootRow model inputs.2.2.1 [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101] [] sm G
      (fun port ↦ fun _ _ ↦ reducePoly modulus_0 8
        ((message 1 (((inputs.2.2.2.2.2.1) * (2)))) * unitPair (tp) port))
      ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1] := by
  apply rootRow_pair_unit
  · simpa only [one_mul, mul_one] using initial_component_row_dio2_state_0 h hmask hpay hG
  · simpa only [one_mul, mul_one] using initial_component_row_dio2_state_1 h hmask hpay hG

end Dio2Proof
