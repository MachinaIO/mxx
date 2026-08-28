import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard050
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard690
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard691
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard753

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound115806
def owner : Owner := ⟨.program ⟨257⟩, ⟨46831⟩⟩
def transferEvent : Nat := 115806
def frameStart : Nat := 115747
def rule : BoundRule := .identity (.predecessor 0 115805 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115805 .coefficient)
      LeftBound115803.bound (LeftBound115803.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound115803.derived selector witness)

def rawBound : CoeffClass := LeftBound115803.bound
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115803.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound115803.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound115806

namespace LeftBound115812
def owner : Owner := ⟨.program ⟨257⟩, ⟨46832⟩⟩
def transferEvent : Nat := 115812
def frameStart : Nat := 115747
def rule : BoundRule := .product (.predecessor 0 115810 .coefficient) (.predecessor 1 115811 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115810 .coefficient)
      LeftAuthority115808.bound (LeftAuthority115808.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events452.exact115809RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority115808.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority115808.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115811 .coefficient)
      LeftBound115806.bound (LeftBound115806.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events452.exact115807RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115806.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115806.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority115808.bound LeftBound115806.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority115808.bound, LeftBound115806.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority115808.actual selector witness) * (LeftBound115806.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound115812

namespace LeftBound115820
def owner : Owner := ⟨.program ⟨257⟩, ⟨46833⟩⟩
def transferEvent : Nat := 115820
def frameStart : Nat := 115747
def rule : BoundRule := .sum [.predecessor 0 115818 .coefficient, .predecessor 1 115819 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115818 .coefficient)
      LeftAuthority115816.bound (LeftAuthority115816.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events452.exact115817RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority115816.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority115816.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115819 .coefficient)
      LeftBound115812.bound (LeftBound115812.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events452.exact115814RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115812.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115812.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority115816.bound, LeftBound115812.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority115816.bound, LeftBound115812.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority115816.actual selector witness, LeftBound115812.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115820

namespace LeftBound115824
def owner : Owner := ⟨.program ⟨257⟩, ⟨47369⟩⟩
def transferEvent : Nat := 115824
def frameStart : Nat := 115747
def rule : BoundRule := .product (.predecessor 0 115822 .coefficient) (.predecessor 1 115823 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115822 .coefficient)
      LeftBound115820.bound (LeftBound115820.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events452.exact115821RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115820.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115820.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115823 .coefficient)
      LeftAuthority115797.bound (LeftAuthority115797.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events452.exact115798RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority115797.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority115797.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound115820.bound LeftAuthority115797.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115820.bound, LeftAuthority115797.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound115820.actual selector witness) * (LeftAuthority115797.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound115824

namespace LeftBound115835
def owner : Owner := ⟨.program ⟨257⟩, ⟨45694⟩⟩
def transferEvent : Nat := 115835
def frameStart : Nat := 115747
def rule : BoundRule := .product (.predecessor 0 115833 .coefficient) (.predecessor 1 115834 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115833 .coefficient)
      LeftAuthority115808.bound (LeftAuthority115808.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events452.exact115809RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority115808.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority115808.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115834 .coefficient)
      LeftAuthority115831.bound (LeftAuthority115831.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events452.exact115832RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority115831.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority115831.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority115808.bound LeftAuthority115831.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority115808.bound, LeftAuthority115831.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority115808.actual selector witness) * (LeftAuthority115831.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound115835

namespace LeftBound115843
def owner : Owner := ⟨.program ⟨257⟩, ⟨45695⟩⟩
def transferEvent : Nat := 115843
def frameStart : Nat := 115747
def rule : BoundRule := .sum [.predecessor 0 115841 .coefficient, .predecessor 1 115842 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115841 .coefficient)
      LeftAuthority115839.bound (LeftAuthority115839.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events452.exact115840RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority115839.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority115839.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115842 .coefficient)
      LeftBound115835.bound (LeftBound115835.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events452.exact115837RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115835.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115835.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority115839.bound, LeftBound115835.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority115839.bound, LeftBound115835.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority115839.actual selector witness, LeftBound115835.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115843

namespace LeftBound115847
def owner : Owner := ⟨.program ⟨257⟩, ⟨47373⟩⟩
def transferEvent : Nat := 115847
def frameStart : Nat := 115747
def rule : BoundRule := .sum [.predecessor 0 115845 .coefficient, .predecessor 1 115846 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115845 .coefficient)
      LeftBound115843.bound (LeftBound115843.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events452.exact115844RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115843.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115843.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115846 .coefficient)
      LeftBound115824.bound (LeftBound115824.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events452.exact115829RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115824.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115824.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound115843.bound, LeftBound115824.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115843.bound, LeftBound115824.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound115843.actual selector witness, LeftBound115824.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115847

namespace LeftBound115860
def owner : Owner := ⟨.program ⟨257⟩, ⟨47371⟩⟩
def transferEvent : Nat := 115860
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 115858 .coefficient, .predecessor 1 115859 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115858 .coefficient)
      LeftBound115689.bound (LeftBound115689.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events452.exact115857RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115689.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115689.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115859 .coefficient)
      LeftBound115672.bound (LeftBound115672.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events451.exact115679RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115672.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115672.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound115689.bound, LeftBound115672.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115689.bound, LeftBound115672.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound115689.actual selector witness, LeftBound115672.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115860

namespace LeftBound115863
def owner : Owner := ⟨.program ⟨257⟩, ⟨47371⟩⟩
def transferEvent : Nat := 115863
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 115857 .summary, .result 115679 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 115857 .summary)
      LeftBound115691.bound (LeftBound115691.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨46235⟩⟩) (rawTerms := some (Proof.Events452.exact115857RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound115691.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 115679 .summary)
      LeftBound115674.bound (LeftBound115674.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨47370⟩⟩) (rawTerms := some (Proof.Events451.exact115679RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound115674.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound115691.bound, LeftBound115674.bound]
def bound : CoeffClass := .finite ⟨32194307824962953452255538577408, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115691.bound, LeftBound115674.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound115691.actual selector witness, LeftBound115674.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115863

namespace LeftBound115867
def owner : Owner := ⟨.program ⟨257⟩, ⟨47372⟩⟩
def transferEvent : Nat := 115867
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 115865 .coefficient) (.predecessor 1 115866 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115865 .coefficient)
      LeftBound115860.bound (LeftBound115860.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events452.exact115864RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115860.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115860.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115866 .coefficient)
      LeftBound15561.bound (LeftBound15561.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15562RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15561.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15561.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound115860.bound LeftBound15561.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115860.bound, LeftBound15561.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound115860.actual selector witness) * (LeftBound15561.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound115867

namespace LeftBound115868
def owner : Owner := ⟨.program ⟨257⟩, ⟨47372⟩⟩
def transferEvent : Nat := 115868
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩ [⟨.result 15558 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15558 .coefficient)
      LeftAuthority15557.bound (LeftAuthority15557.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7151⟩⟩) (rawTerms := some (Proof.Events060.exact15558RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15557.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15557.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15557.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15557.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15557.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound115868

namespace LeftBound115869
def owner : Owner := ⟨.program ⟨257⟩, ⟨47372⟩⟩
def transferEvent : Nat := 115869
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 115864 .summary) (.transfer 115868) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 115864 .summary)
      LeftBound115863.bound (LeftBound115863.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨47371⟩⟩) (rawTerms := some (Proof.Events452.exact115864RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound115863.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 115868)
      LeftBound115868.bound (LeftBound115868.actual selector witness) := by
  exact .transfer (LeftBound115868.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound115863.bound LeftBound115868.bound
def bound : CoeffClass := .finite ⟨345683748063931943722519589062084311121920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115863.bound, LeftBound115868.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound115863.actual selector witness) * (LeftBound115868.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound115869

namespace LeftBound115884
def owner : Owner := ⟨.program ⟨257⟩, ⟨44690⟩⟩
def transferEvent : Nat := 115884
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 115882 .coefficient) (.predecessor 1 115883 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115882 .coefficient)
      LeftBound106391.bound (LeftBound106391.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106395RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106391.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106391.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115883 .coefficient)
      LeftAuthority115880.bound (LeftAuthority115880.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events452.exact115881RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority115880.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority115880.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound106391.bound LeftAuthority115880.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106391.bound, LeftAuthority115880.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound106391.actual selector witness) * (LeftAuthority115880.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound115884

namespace LeftBound115885
def owner : Owner := ⟨.program ⟨257⟩, ⟨44690⟩⟩
def transferEvent : Nat := 115885
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨44688⟩⟩]⟩ [⟨.result 115881 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 115881 .coefficient)
      LeftAuthority115880.bound (LeftAuthority115880.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨44688⟩⟩) (rawTerms := some (Proof.Events452.exact115881RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority115880.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority115880.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority115880.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority115880.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority115880.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound115885

namespace LeftBound115886
def owner : Owner := ⟨.program ⟨257⟩, ⟨44690⟩⟩
def transferEvent : Nat := 115886
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 106395 .summary) (.transfer 115885) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 106395 .summary)
      LeftBound106394.bound (LeftBound106394.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44312⟩⟩) (rawTerms := some (Proof.Events415.exact106395RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound106394.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 115885)
      LeftBound115885.bound (LeftBound115885.actual selector witness) := by
  exact .transfer (LeftBound115885.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound106394.bound LeftBound115885.bound
def bound : CoeffClass := .finite ⟨32193718473625689247691015454720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106394.bound, LeftBound115885.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound106394.actual selector witness) * (LeftBound115885.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound115886

namespace LeftBound115897
def owner : Owner := ⟨.program ⟨257⟩, ⟨43554⟩⟩
def transferEvent : Nat := 115897
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 115895 .coefficient) (.value (.predecessor 1 115896 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115895 .coefficient)
      LeftAuthority115893.bound (LeftAuthority115893.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events452.exact115894RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority115893.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority115893.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115896 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority115893.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority115893.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority115893.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound115897

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
