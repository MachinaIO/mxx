import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard074
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard075
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1388
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1391
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1397

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound208422
def owner : Owner := ⟨.program ⟨257⟩, ⟨46826⟩⟩
def transferEvent : Nat := 208422
def frameStart : Nat := 208366
def rule : BoundRule := .sum [.predecessor 0 208420 .coefficient, .predecessor 1 208421 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 208420 .coefficient)
      LeftBound208405.bound (LeftBound208405.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound208405.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 208421 .coefficient)
      LeftAuthority208418.bound (LeftAuthority208418.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority208418.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound208405.bound, LeftAuthority208418.bound]
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound208405.bound, LeftAuthority208418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound208405.actual selector witness, LeftAuthority208418.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound208422

namespace LeftBound208425
def owner : Owner := ⟨.program ⟨257⟩, ⟨46827⟩⟩
def transferEvent : Nat := 208425
def frameStart : Nat := 208366
def rule : BoundRule := .identity (.predecessor 0 208424 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 208424 .coefficient)
      LeftBound208422.bound (LeftBound208422.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound208422.derived selector witness)

def rawBound : CoeffClass := LeftBound208422.bound
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound208422.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound208422.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound208425

namespace LeftBound208431
def owner : Owner := ⟨.program ⟨257⟩, ⟨46828⟩⟩
def transferEvent : Nat := 208431
def frameStart : Nat := 208366
def rule : BoundRule := .product (.predecessor 0 208429 .coefficient) (.predecessor 1 208430 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 208429 .coefficient)
      LeftAuthority208427.bound (LeftAuthority208427.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events814.exact208428RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority208427.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority208427.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 208430 .coefficient)
      LeftBound208425.bound (LeftBound208425.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events814.exact208426RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound208425.bound, RecordedBoundRefines] <;> decide)
      (LeftBound208425.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority208427.bound LeftBound208425.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority208427.bound, LeftBound208425.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority208427.actual selector witness) * (LeftBound208425.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound208431

namespace LeftBound208439
def owner : Owner := ⟨.program ⟨257⟩, ⟨46829⟩⟩
def transferEvent : Nat := 208439
def frameStart : Nat := 208366
def rule : BoundRule := .sum [.predecessor 0 208437 .coefficient, .predecessor 1 208438 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 208437 .coefficient)
      LeftAuthority208435.bound (LeftAuthority208435.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events814.exact208436RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority208435.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority208435.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 208438 .coefficient)
      LeftBound208431.bound (LeftBound208431.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events814.exact208433RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound208431.bound, RecordedBoundRefines] <;> decide)
      (LeftBound208431.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority208435.bound, LeftBound208431.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority208435.bound, LeftBound208431.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority208435.actual selector witness, LeftBound208431.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound208439

namespace LeftBound208443
def owner : Owner := ⟨.program ⟨257⟩, ⟨47350⟩⟩
def transferEvent : Nat := 208443
def frameStart : Nat := 208366
def rule : BoundRule := .product (.predecessor 0 208441 .coefficient) (.predecessor 1 208442 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 208441 .coefficient)
      LeftBound208439.bound (LeftBound208439.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events814.exact208440RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound208439.bound, RecordedBoundRefines] <;> decide)
      (LeftBound208439.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 208442 .coefficient)
      LeftAuthority208416.bound (LeftAuthority208416.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events814.exact208417RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority208416.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority208416.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound208439.bound LeftAuthority208416.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound208439.bound, LeftAuthority208416.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound208439.actual selector witness) * (LeftAuthority208416.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound208443

namespace LeftBound208454
def owner : Owner := ⟨.program ⟨257⟩, ⟨45684⟩⟩
def transferEvent : Nat := 208454
def frameStart : Nat := 208366
def rule : BoundRule := .product (.predecessor 0 208452 .coefficient) (.predecessor 1 208453 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 208452 .coefficient)
      LeftAuthority208427.bound (LeftAuthority208427.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events814.exact208428RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority208427.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority208427.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 208453 .coefficient)
      LeftAuthority208450.bound (LeftAuthority208450.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events814.exact208451RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority208450.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority208450.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority208427.bound LeftAuthority208450.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority208427.bound, LeftAuthority208450.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority208427.actual selector witness) * (LeftAuthority208450.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound208454

namespace LeftBound208462
def owner : Owner := ⟨.program ⟨257⟩, ⟨45685⟩⟩
def transferEvent : Nat := 208462
def frameStart : Nat := 208366
def rule : BoundRule := .sum [.predecessor 0 208460 .coefficient, .predecessor 1 208461 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 208460 .coefficient)
      LeftAuthority208458.bound (LeftAuthority208458.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events814.exact208459RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority208458.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority208458.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 208461 .coefficient)
      LeftBound208454.bound (LeftBound208454.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events814.exact208456RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound208454.bound, RecordedBoundRefines] <;> decide)
      (LeftBound208454.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority208458.bound, LeftBound208454.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority208458.bound, LeftBound208454.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority208458.actual selector witness, LeftBound208454.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound208462

namespace LeftBound208466
def owner : Owner := ⟨.program ⟨257⟩, ⟨47353⟩⟩
def transferEvent : Nat := 208466
def frameStart : Nat := 208366
def rule : BoundRule := .sum [.predecessor 0 208464 .coefficient, .predecessor 1 208465 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 208464 .coefficient)
      LeftBound208462.bound (LeftBound208462.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events814.exact208463RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound208462.bound, RecordedBoundRefines] <;> decide)
      (LeftBound208462.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 208465 .coefficient)
      LeftBound208443.bound (LeftBound208443.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events814.exact208448RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound208443.bound, RecordedBoundRefines] <;> decide)
      (LeftBound208443.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound208462.bound, LeftBound208443.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound208462.bound, LeftBound208443.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound208462.actual selector witness, LeftBound208443.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound208466

namespace LeftBound208479
def owner : Owner := ⟨.program ⟨257⟩, ⟨47352⟩⟩
def transferEvent : Nat := 208479
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 208477 .coefficient, .predecessor 1 208478 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 208477 .coefficient)
      LeftBound208308.bound (LeftBound208308.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events814.exact208476RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound208308.bound, RecordedBoundRefines] <;> decide)
      (LeftBound208308.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 208478 .coefficient)
      LeftBound208291.bound (LeftBound208291.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events813.exact208298RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound208291.bound, RecordedBoundRefines] <;> decide)
      (LeftBound208291.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound208308.bound, LeftBound208291.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound208308.bound, LeftBound208291.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound208308.actual selector witness, LeftBound208291.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound208479

namespace LeftBound208482
def owner : Owner := ⟨.program ⟨257⟩, ⟨47352⟩⟩
def transferEvent : Nat := 208482
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 208476 .summary, .result 208298 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 208476 .summary)
      LeftBound208310.bound (LeftBound208310.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨46219⟩⟩) (rawTerms := some (Proof.Events814.exact208476RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound208310.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 208298 .summary)
      LeftBound208293.bound (LeftBound208293.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨47351⟩⟩) (rawTerms := some (Proof.Events813.exact208298RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound208293.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound208310.bound, LeftBound208293.bound]
def bound : CoeffClass := .finite ⟨32194307824962953452255538577408, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound208310.bound, LeftBound208293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound208310.actual selector witness, LeftBound208293.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound208482

namespace LeftBound208506
def owner : Owner := ⟨.program ⟨257⟩, ⟨42477⟩⟩
def transferEvent : Nat := 208506
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 208504 .coefficient) (.predecessor 1 208505 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 208504 .coefficient)
      LeftAuthority9863.bound (LeftAuthority9863.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9864RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9863.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9863.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 208505 .coefficient)
      LeftBound207526.bound (LeftBound207526.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events810.exact207528RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207526.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207526.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority9863.bound LeftBound207526.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9863.bound, LeftBound207526.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority9863.actual selector witness) * (LeftBound207526.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound208506

namespace LeftBound208511
def owner : Owner := ⟨.program ⟨257⟩, ⟨8589⟩⟩
def transferEvent : Nat := 208511
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 208509 .coefficient) (.predecessor 1 208510 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 208509 .coefficient)
      LeftBound207397.bound (LeftBound207397.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events810.exact207398RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207397.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207397.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 208510 .coefficient)
      LeftBound18081.bound (LeftBound18081.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events070.exact18082RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18081.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18081.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound207397.bound LeftBound18081.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207397.bound, LeftBound18081.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound207397.actual selector witness) * (LeftBound18081.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound208511

namespace LeftBound208516
def owner : Owner := ⟨.program ⟨257⟩, ⟨42478⟩⟩
def transferEvent : Nat := 208516
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 208514 .coefficient, .predecessor 1 208515 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 208514 .coefficient)
      LeftBound208511.bound (LeftBound208511.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events814.exact208513RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound208511.bound, RecordedBoundRefines] <;> decide)
      (LeftBound208511.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 208515 .coefficient)
      LeftBound208506.bound (LeftBound208506.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events814.exact208508RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound208506.bound, RecordedBoundRefines] <;> decide)
      (LeftBound208506.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound208511.bound, LeftBound208506.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound208511.bound, LeftBound208506.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound208511.actual selector witness, LeftBound208506.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound208516

namespace LeftBound208520
def owner : Owner := ⟨.program ⟨257⟩, ⟨42479⟩⟩
def transferEvent : Nat := 208520
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 208518 .coefficient, .predecessor 1 208519 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 208518 .coefficient)
      LeftBound208516.bound (LeftBound208516.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events814.exact208517RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound208516.bound, RecordedBoundRefines] <;> decide)
      (LeftBound208516.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 208519 .coefficient)
      LeftBound18073.bound (LeftBound18073.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events070.exact18074RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18073.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18073.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound208516.bound, LeftBound18073.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound208516.bound, LeftBound18073.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound208516.actual selector witness, LeftBound18073.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound208520

namespace LeftBound208521
def owner : Owner := ⟨.program ⟨257⟩, ⟨42479⟩⟩
def transferEvent : Nat := 208521
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨109⟩⟩]⟩ [⟨.result 18074 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 18074 .coefficient)
      LeftBound18073.bound (LeftBound18073.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨109⟩⟩) (rawTerms := some (Proof.Events070.exact18074RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18073.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18073.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound18073.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18073.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound18073.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound208521

namespace LeftBound208526
def owner : Owner := ⟨.program ⟨257⟩, ⟨42480⟩⟩
def transferEvent : Nat := 208526
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 208524 .coefficient) (.predecessor 1 208525 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 208524 .coefficient)
      LeftBound208520.bound (LeftBound208520.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events814.exact208523RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound208520.bound, RecordedBoundRefines] <;> decide)
      (LeftBound208520.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 208525 .coefficient)
      LeftAuthority9866.bound (LeftAuthority9866.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9867RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9866.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9866.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound208520.bound LeftAuthority9866.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound208520.bound, LeftAuthority9866.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound208520.actual selector witness) * (LeftAuthority9866.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound208526

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
