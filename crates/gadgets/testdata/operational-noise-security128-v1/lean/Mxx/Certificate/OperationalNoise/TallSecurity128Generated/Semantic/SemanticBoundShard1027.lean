import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard986
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1026

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound154427
def owner : Owner := ⟨.program ⟨257⟩, ⟨57382⟩⟩
def transferEvent : Nat := 154427
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 149120 .summary) (.transfer 154426) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 149120 .summary)
      LeftBound149118.bound (LeftBound149118.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5545⟩⟩) (rawTerms := some (Proof.Events582.exact149120RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound149118.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 154426)
      LeftBound154426.bound (LeftBound154426.actual selector witness) := by
  exact .transfer (LeftBound154426.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound149118.bound LeftBound154426.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound149118.bound, LeftBound154426.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound149118.actual selector witness) * (LeftBound154426.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound154427

namespace LeftBound154506
def owner : Owner := ⟨.program ⟨257⟩, ⟨56425⟩⟩
def transferEvent : Nat := 154506
def frameStart : Nat := 154477
def rule : BoundRule := .product (.predecessor 0 154504 .coefficient) (.predecessor 1 154505 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 154504 .coefficient)
      LeftAuthority154502.bound (LeftAuthority154502.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events603.exact154503RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority154502.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority154502.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 154505 .coefficient)
      LeftAuthority154499.bound (LeftAuthority154499.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events603.exact154500RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority154499.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority154499.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority154502.bound LeftAuthority154499.bound
def bound : CoeffClass := .finite ⟨256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority154502.bound, LeftAuthority154499.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority154502.actual selector witness) * (LeftAuthority154499.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound154506

namespace LeftBound154510
def owner : Owner := ⟨.program ⟨257⟩, ⟨56426⟩⟩
def transferEvent : Nat := 154510
def frameStart : Nat := 154477
def rule : BoundRule := .identity (.predecessor 0 154509 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 154509 .coefficient)
      LeftBound154506.bound (LeftBound154506.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events603.exact154508RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound154506.bound, RecordedBoundRefines] <;> decide)
      (LeftBound154506.derived selector witness)

def rawBound : CoeffClass := LeftBound154506.bound
def bound : CoeffClass := .finite ⟨256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound154506.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound154506.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound154510

namespace LeftBound154527
def owner : Owner := ⟨.program ⟨257⟩, ⟨58234⟩⟩
def transferEvent : Nat := 154527
def frameStart : Nat := 154477
def rule : BoundRule := .sum [.predecessor 0 154525 .coefficient, .predecessor 1 154526 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 154525 .coefficient)
      LeftBound154510.bound (LeftBound154510.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound154510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 154526 .coefficient)
      LeftAuthority154523.bound (LeftAuthority154523.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority154523.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound154510.bound, LeftAuthority154523.bound]
def bound : CoeffClass := .finite ⟨256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound154510.bound, LeftAuthority154523.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound154510.actual selector witness, LeftAuthority154523.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound154527

namespace LeftBound154530
def owner : Owner := ⟨.program ⟨257⟩, ⟨58235⟩⟩
def transferEvent : Nat := 154530
def frameStart : Nat := 154477
def rule : BoundRule := .identity (.predecessor 0 154529 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 154529 .coefficient)
      LeftBound154527.bound (LeftBound154527.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound154527.derived selector witness)

def rawBound : CoeffClass := LeftBound154527.bound
def bound : CoeffClass := .finite ⟨256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound154527.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound154527.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound154530

namespace LeftBound154536
def owner : Owner := ⟨.program ⟨257⟩, ⟨58236⟩⟩
def transferEvent : Nat := 154536
def frameStart : Nat := 154477
def rule : BoundRule := .product (.predecessor 0 154534 .coefficient) (.predecessor 1 154535 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 154534 .coefficient)
      LeftAuthority154532.bound (LeftAuthority154532.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events603.exact154533RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority154532.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority154532.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 154535 .coefficient)
      LeftBound154530.bound (LeftBound154530.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events603.exact154531RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound154530.bound, RecordedBoundRefines] <;> decide)
      (LeftBound154530.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority154532.bound LeftBound154530.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority154532.bound, LeftBound154530.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority154532.actual selector witness) * (LeftBound154530.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound154536

namespace LeftBound154552
def owner : Owner := ⟨.program ⟨257⟩, ⟨9533⟩⟩
def transferEvent : Nat := 154552
def frameStart : Nat := 154477
def rule : BoundRule := .scale (.predecessor 0 154550 .coefficient) (.value (.predecessor 1 154551 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 154550 .coefficient)
      LeftAuthority154548.bound (LeftAuthority154548.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events603.exact154549RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority154548.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority154548.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 154551 .coefficient)
      LeftAuthority154539.bound (LeftAuthority154539.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority154539.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority154548.bound LeftAuthority154539.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority154548.bound, LeftAuthority154539.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority154548.actual selector witness) * (LeftAuthority154539.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound154552

namespace LeftBound154555
def owner : Owner := ⟨.program ⟨257⟩, ⟨7290⟩⟩
def transferEvent : Nat := 154555
def frameStart : Nat := 154477
def rule : BoundRule := .identity (.predecessor 0 154554 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 154554 .coefficient)
      LeftAuthority154542.bound (LeftAuthority154542.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events603.exact154543RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority154542.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority154542.derived selector witness)

def rawBound : CoeffClass := LeftAuthority154542.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority154542.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority154542.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound154555

namespace LeftBound154559
def owner : Owner := ⟨.program ⟨257⟩, ⟨9534⟩⟩
def transferEvent : Nat := 154559
def frameStart : Nat := 154477
def rule : BoundRule := .product (.predecessor 0 154557 .coefficient) (.predecessor 1 154558 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 154557 .coefficient)
      LeftBound154555.bound (LeftBound154555.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events603.exact154556RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound154555.bound, RecordedBoundRefines] <;> decide)
      (LeftBound154555.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 154558 .coefficient)
      LeftBound154552.bound (LeftBound154552.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events603.exact154553RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound154552.bound, RecordedBoundRefines] <;> decide)
      (LeftBound154552.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound154555.bound LeftBound154552.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound154555.bound, LeftBound154552.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound154555.actual selector witness) * (LeftBound154552.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound154559

namespace LeftBound154564
def owner : Owner := ⟨.program ⟨257⟩, ⟨58237⟩⟩
def transferEvent : Nat := 154564
def frameStart : Nat := 154477
def rule : BoundRule := .sum [.predecessor 0 154562 .coefficient, .predecessor 1 154563 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 154562 .coefficient)
      LeftBound154559.bound (LeftBound154559.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events603.exact154561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound154559.bound, RecordedBoundRefines] <;> decide)
      (LeftBound154559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 154563 .coefficient)
      LeftBound154536.bound (LeftBound154536.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events603.exact154538RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound154536.bound, RecordedBoundRefines] <;> decide)
      (LeftBound154536.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound154559.bound, LeftBound154536.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound154559.bound, LeftBound154536.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound154559.actual selector witness, LeftBound154536.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound154564

namespace LeftBound154568
def owner : Owner := ⟨.program ⟨257⟩, ⟨58449⟩⟩
def transferEvent : Nat := 154568
def frameStart : Nat := 154477
def rule : BoundRule := .product (.predecessor 0 154566 .coefficient) (.predecessor 1 154567 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 154566 .coefficient)
      LeftBound154564.bound (LeftBound154564.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events603.exact154565RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound154564.bound, RecordedBoundRefines] <;> decide)
      (LeftBound154564.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 154567 .coefficient)
      LeftAuthority154521.bound (LeftAuthority154521.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events603.exact154522RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority154521.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority154521.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound154564.bound LeftAuthority154521.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound154564.bound, LeftAuthority154521.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound154564.actual selector witness) * (LeftAuthority154521.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound154568

namespace LeftBound154579
def owner : Owner := ⟨.program ⟨257⟩, ⟨56826⟩⟩
def transferEvent : Nat := 154579
def frameStart : Nat := 154477
def rule : BoundRule := .product (.predecessor 0 154577 .coefficient) (.predecessor 1 154578 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 154577 .coefficient)
      LeftAuthority154532.bound (LeftAuthority154532.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events603.exact154533RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority154532.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority154532.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 154578 .coefficient)
      LeftAuthority154575.bound (LeftAuthority154575.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events603.exact154576RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority154575.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority154575.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority154532.bound LeftAuthority154575.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority154532.bound, LeftAuthority154575.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority154532.actual selector witness) * (LeftAuthority154575.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound154579

namespace LeftBound154587
def owner : Owner := ⟨.program ⟨257⟩, ⟨56827⟩⟩
def transferEvent : Nat := 154587
def frameStart : Nat := 154477
def rule : BoundRule := .sum [.predecessor 0 154585 .coefficient, .predecessor 1 154586 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 154585 .coefficient)
      LeftAuthority154583.bound (LeftAuthority154583.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events603.exact154584RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority154583.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority154583.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 154586 .coefficient)
      LeftBound154579.bound (LeftBound154579.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events603.exact154581RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound154579.bound, RecordedBoundRefines] <;> decide)
      (LeftBound154579.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority154583.bound, LeftBound154579.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority154583.bound, LeftBound154579.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority154583.actual selector witness, LeftBound154579.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound154587

namespace LeftBound154591
def owner : Owner := ⟨.program ⟨257⟩, ⟨58450⟩⟩
def transferEvent : Nat := 154591
def frameStart : Nat := 154477
def rule : BoundRule := .sum [.predecessor 0 154589 .coefficient, .predecessor 1 154590 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 154589 .coefficient)
      LeftBound154587.bound (LeftBound154587.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events603.exact154588RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound154587.bound, RecordedBoundRefines] <;> decide)
      (LeftBound154587.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 154590 .coefficient)
      LeftBound154568.bound (LeftBound154568.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events603.exact154573RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound154568.bound, RecordedBoundRefines] <;> decide)
      (LeftBound154568.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound154587.bound, LeftBound154568.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound154587.bound, LeftBound154568.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound154587.actual selector witness, LeftBound154568.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound154591

namespace LeftBound154604
def owner : Owner := ⟨.program ⟨257⟩, ⟨58448⟩⟩
def transferEvent : Nat := 154604
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 154602 .coefficient, .predecessor 1 154603 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 154602 .coefficient)
      LeftBound154425.bound (LeftBound154425.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events603.exact154601RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound154425.bound, RecordedBoundRefines] <;> decide)
      (LeftBound154425.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 154603 .coefficient)
      LeftBound154408.bound (LeftBound154408.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events603.exact154415RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound154408.bound, RecordedBoundRefines] <;> decide)
      (LeftBound154408.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound154425.bound, LeftBound154408.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound154425.bound, LeftBound154408.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound154425.actual selector witness, LeftBound154408.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound154604

namespace LeftBound154607
def owner : Owner := ⟨.program ⟨257⟩, ⟨58448⟩⟩
def transferEvent : Nat := 154607
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 154601 .summary, .result 154415 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 154601 .summary)
      LeftBound154427.bound (LeftBound154427.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨57382⟩⟩) (rawTerms := some (Proof.Events603.exact154601RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound154427.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 154415 .summary)
      LeftBound154410.bound (LeftBound154410.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58447⟩⟩) (rawTerms := some (Proof.Events603.exact154415RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound154410.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound154427.bound, LeftBound154410.bound]
def bound : CoeffClass := .finite ⟨2997944351807545540608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound154427.bound, LeftBound154410.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound154427.actual selector witness, LeftBound154410.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound154607

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
