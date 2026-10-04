import MxxRuntime

/-!
Concrete regular gadget layouts, a multi-tower one and one with a dropped tower, and the
`MxxRuntime` facts generated relations rely on, checked on them: reconstruction, digit bounds, the
backend lookup, and the approximate reconstruction of a layout that drops towers.
-/

namespace ExampleBackend

noncomputable section

def moduli0 : List Nat := [1021]
def layout0 : MxxRuntime.RegularLayout 1021 :=
  { crtModuli := moduli0
    droppedModuli := 0
    dropped_lt := by decide
    crtModuli_nonempty := by decide
    modulus_pos := by decide
    pairwise_coprime := by unfold Pairwise; decide
    product_eq := by decide
    baseBits := 5
    base := 32
    base_eq := by norm_num
    base_gt_one := by norm_num
    base_even := by decide
    digitsPerTower := 2
    digits_pos := by norm_num
    capacity := by decide }

def moduli1 : List Nat := [1021, 1013]
def layout1 : MxxRuntime.RegularLayout 1034273 :=
  { crtModuli := moduli1
    droppedModuli := 0
    dropped_lt := by decide
    crtModuli_nonempty := by decide
    modulus_pos := by decide
    pairwise_coprime := by unfold Pairwise; decide
    product_eq := by decide
    baseBits := 5
    base := 32
    base_eq := by norm_num
    base_gt_one := by norm_num
    base_even := by decide
    digitsPerTower := 2
    digits_pos := by norm_num
    capacity := by decide }

def moduli2 : List Nat := [1021, 1013, 1009]
def layout2 : MxxRuntime.RegularLayout 1043581457 :=
  { crtModuli := moduli2
    droppedModuli := 1
    dropped_lt := by decide
    crtModuli_nonempty := by decide
    modulus_pos := by decide
    pairwise_coprime := by unfold Pairwise; decide
    product_eq := by decide
    baseBits := 5
    base := 32
    base_eq := by norm_num
    base_gt_one := by norm_num
    base_even := by decide
    digitsPerTower := 2
    digits_pos := by norm_num
    capacity := by decide }

def backend : MxxRuntime.BackendContext where
  regularLayout q n :=
    if h : q = 1021 ∧ n = 2 then
      some (h.1.symm ▸ layout0)
    else
    if h : q = 1034273 ∧ n = 2 then
      some (h.1.symm ▸ layout1)
    else
    if h : q = 1043581457 ∧ n = 2 then
      some (h.1.symm ▸ layout2)
    else
    none

end
end ExampleBackend


namespace ExampleConcreteRegular

noncomputable section

def q : Nat := 1021
def ringDimension : Nat := 2
def base : Nat := 32
def digitsPerTower : Nat := 2
def regularDigits : Nat := 2
def moduli : List Nat := [1021]

def concreteLayout : MxxRuntime.RegularLayout q :=
  { crtModuli := moduli
    crtModuli_nonempty := by simp [moduli]
    droppedModuli := 0
    dropped_lt := by norm_num [moduli]
    modulus_pos := by
      intro tower; fin_cases tower; norm_num [moduli]
    pairwise_coprime := by simp [moduli]
    product_eq := by norm_num [q, moduli]
    baseBits := 5
    base := base
    base_eq := by norm_num [base]
    base_gt_one := by norm_num [base]
    base_even := by refine ⟨16, by norm_num [base]⟩
    digitsPerTower := digitsPerTower
    digits_pos := by norm_num [digitsPerTower]
    capacity := by
      intro tower; fin_cases tower; norm_num [moduli, base, digitsPerTower] }

def publicMatrix : Mxx.Primitives.ExactMatrix q ringDimension 1 2 :=
  MxxRuntime.regularGadgetMatrix concreteLayout
def target : Mxx.Primitives.ExactMatrix q ringDimension 1 1 := fun _ _ => 1
def trapdoor : MxxRuntime.TrapdoorValue
    (Mxx.Primitives.ExactMatrix q ringDimension 1 2) Unit :=
  MxxRuntime.regularGadgetTrapdoor concreteLayout 0

def preimage : Mxx.Primitives.ExactMatrix q ringDimension 2 1 :=
  MxxRuntime.regularDecomposeMatrix concreteLayout target

theorem generated_layout_capacity :
    ∀ tower : Fin moduli.length, moduli.get tower ≤ base ^ digitsPerTower := by
  exact concreteLayout.capacity

theorem generated_digit_bound
    (value : Mxx.Primitives.ExactPoly q ringDimension)
    (limb : MxxRuntime.RegularLimb concreteLayout)
    (coefficient : Fin ringDimension) :
    (MxxRuntime.regularDigitCoefficient concreteLayout value limb coefficient).natAbs ≤ 16 := by
  simpa [MxxRuntime.regularDigitCoefficient, base, concreteLayout] using
    (Mxx.Primitives.balancedDigit_abs_le concreteLayout.base
      concreteLayout.base_gt_one concreteLayout.base_even _)

theorem generated_public_preimage_fixture :
    MxxRuntime.publicGadgetPreimageRuns publicMatrix trapdoor target preimage := by
  exact MxxRuntime.regularGadgetTrapdoor_preimage concreteLayout 0 target (by rfl)

theorem generated_arbitrary_target_reconstruction
    (value : Mxx.Primitives.ExactMatrix q ringDimension 1 1) :
    MxxRuntime.regularGadgetMatrix (n := ringDimension) (rows := 1) concreteLayout *
      MxxRuntime.regularDecomposeMatrix concreteLayout value = value := by
  exact MxxRuntime.regularGadgetMatrix_reconstruct concreteLayout value
    (by norm_num [q]) (by norm_num [ringDimension]) (by rfl)

theorem generated_preimage_bound : Mxx.Primitives.PreimageWithin preimage 16 := by
  exact MxxRuntime.regularDecomposeMatrix_bounded concreteLayout target
    (by norm_num [q]) (by norm_num [ringDimension])

theorem generated_backend_lookup :
    ExampleBackend.backend.regularLayout q ringDimension =
      some ExampleBackend.layout0 := by
  simp [ExampleBackend.backend, q, ringDimension]

theorem generated_backend_decomposition
    (value : Mxx.Primitives.ExactMatrix q ringDimension 1 1) :
    MxxRuntime.gadgetDecomposeRuns ExampleBackend.backend base regularDigits value
      (MxxRuntime.regularDecomposeMatrix ExampleBackend.layout0 value) := by
  refine ⟨ExampleBackend.layout0, generated_backend_lookup, ?_, ?_, rfl, ?_, ?_⟩
  · norm_num [base, ExampleBackend.layout0]
  · norm_num [regularDigits, ExampleBackend.layout0, ExampleBackend.moduli0,
      MxxRuntime.RegularLayout.digitCount, MxxRuntime.RegularLayout.retainedTowers]
  · simp [MxxRuntime.castMatrixRows]
  · exact MxxRuntime.regularGadgetMatrix_residual_exact _ _
      (by norm_num [q]) (by norm_num [ringDimension]) (by rfl)

theorem generated_multitower_lookup :
    ExampleBackend.backend.regularLayout 1034273 2 =
      some ExampleBackend.layout1 := by
  simp [ExampleBackend.backend]

theorem generated_multitower_reconstruction
    (value : Mxx.Primitives.ExactMatrix 1034273 2 1 1) :
    MxxRuntime.regularGadgetMatrix (n := 2) (rows := 1) ExampleBackend.layout1 *
      MxxRuntime.regularDecomposeMatrix ExampleBackend.layout1 value = value := by
  exact MxxRuntime.regularGadgetMatrix_reconstruct ExampleBackend.layout1 value
    (by norm_num) (by norm_num) (by rfl)

end
end ExampleConcreteRegular

namespace ExampleApproximateRegular
open Mxx.Primitives MxxRuntime

theorem generated_approximate_shape : ExampleBackend.layout2.digitCount = 4 := by decide
theorem generated_approximate_bound : ExampleBackend.layout2.errorBound = 504 :=
  by decide

theorem generated_approximate_digits
    (target : ExactMatrix 1043581457 2 1 3) :
    PreimageWithin (regularDecomposeMatrix ExampleBackend.layout2 target) 16 := by
  exact regularDecomposeMatrix_bounded _ _ (by decide) (by decide)

/-- The bounded primitive premise is tied to the exact corrected decomposition of this target. -/
theorem generated_approximate_reconstruction
    (target : ExactMatrix 1043581457 2 1 3)
    (hresidual : PreimageWithin
      (target - regularGadgetMatrix ExampleBackend.layout2 *
        regularDecomposeMatrix ExampleBackend.layout2 target) 504) :
    Approx target (regularGadgetMatrix ExampleBackend.layout2 *
      regularDecomposeMatrix ExampleBackend.layout2 target) 504 := by
  exact regularGadgetMatrix_approximate _ _ hresidual

end ExampleApproximateRegular
