import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard885
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard914

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound138465
def owner : Owner := ⟨.program ⟨257⟩, ⟨68901⟩⟩
def transferEvent : Nat := 138465
def frameStart : Nat := 138406
def rule : BoundRule := .product (.predecessor 0 138463 .coefficient) (.predecessor 1 138464 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 138463 .coefficient)
      LeftAuthority138461.bound (LeftAuthority138461.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events540.exact138462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority138461.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority138461.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 138464 .coefficient)
      LeftBound138459.bound (LeftBound138459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events540.exact138460RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound138459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound138459.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority138461.bound LeftBound138459.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority138461.bound, LeftBound138459.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority138461.actual selector witness) * (LeftBound138459.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound138465

namespace LeftBound138481
def owner : Owner := ⟨.program ⟨257⟩, ⟨9542⟩⟩
def transferEvent : Nat := 138481
def frameStart : Nat := 138406
def rule : BoundRule := .scale (.predecessor 0 138479 .coefficient) (.value (.predecessor 1 138480 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 138479 .coefficient)
      LeftAuthority138477.bound (LeftAuthority138477.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events540.exact138478RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority138477.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority138477.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 138480 .coefficient)
      LeftAuthority138468.bound (LeftAuthority138468.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority138468.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority138477.bound LeftAuthority138468.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority138477.bound, LeftAuthority138468.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority138477.actual selector witness) * (LeftAuthority138468.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound138481

namespace LeftBound138484
def owner : Owner := ⟨.program ⟨257⟩, ⟨7294⟩⟩
def transferEvent : Nat := 138484
def frameStart : Nat := 138406
def rule : BoundRule := .identity (.predecessor 0 138483 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 138483 .coefficient)
      LeftAuthority138471.bound (LeftAuthority138471.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events540.exact138472RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority138471.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority138471.derived selector witness)

def rawBound : CoeffClass := LeftAuthority138471.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority138471.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority138471.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound138484

namespace LeftBound138488
def owner : Owner := ⟨.program ⟨257⟩, ⟨9543⟩⟩
def transferEvent : Nat := 138488
def frameStart : Nat := 138406
def rule : BoundRule := .product (.predecessor 0 138486 .coefficient) (.predecessor 1 138487 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 138486 .coefficient)
      LeftBound138484.bound (LeftBound138484.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events540.exact138485RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound138484.bound, RecordedBoundRefines] <;> decide)
      (LeftBound138484.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 138487 .coefficient)
      LeftBound138481.bound (LeftBound138481.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events540.exact138482RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound138481.bound, RecordedBoundRefines] <;> decide)
      (LeftBound138481.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound138484.bound LeftBound138481.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound138484.bound, LeftBound138481.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound138484.actual selector witness) * (LeftBound138481.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound138488

namespace LeftBound138493
def owner : Owner := ⟨.program ⟨257⟩, ⟨68902⟩⟩
def transferEvent : Nat := 138493
def frameStart : Nat := 138406
def rule : BoundRule := .sum [.predecessor 0 138491 .coefficient, .predecessor 1 138492 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 138491 .coefficient)
      LeftBound138488.bound (LeftBound138488.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events540.exact138490RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound138488.bound, RecordedBoundRefines] <;> decide)
      (LeftBound138488.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 138492 .coefficient)
      LeftBound138465.bound (LeftBound138465.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events540.exact138467RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound138465.bound, RecordedBoundRefines] <;> decide)
      (LeftBound138465.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound138488.bound, LeftBound138465.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound138488.bound, LeftBound138465.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound138488.actual selector witness, LeftBound138465.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound138493

namespace LeftBound138497
def owner : Owner := ⟨.program ⟨257⟩, ⟨69166⟩⟩
def transferEvent : Nat := 138497
def frameStart : Nat := 138406
def rule : BoundRule := .product (.predecessor 0 138495 .coefficient) (.predecessor 1 138496 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 138495 .coefficient)
      LeftBound138493.bound (LeftBound138493.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events540.exact138494RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound138493.bound, RecordedBoundRefines] <;> decide)
      (LeftBound138493.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 138496 .coefficient)
      LeftAuthority138450.bound (LeftAuthority138450.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events540.exact138451RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority138450.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority138450.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound138493.bound LeftAuthority138450.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound138493.bound, LeftAuthority138450.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound138493.actual selector witness) * (LeftAuthority138450.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound138497

namespace LeftBound138508
def owner : Owner := ⟨.program ⟨257⟩, ⟨65734⟩⟩
def transferEvent : Nat := 138508
def frameStart : Nat := 138406
def rule : BoundRule := .product (.predecessor 0 138506 .coefficient) (.predecessor 1 138507 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 138506 .coefficient)
      LeftAuthority138461.bound (LeftAuthority138461.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events540.exact138462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority138461.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority138461.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 138507 .coefficient)
      LeftAuthority138504.bound (LeftAuthority138504.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events541.exact138505RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority138504.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority138504.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority138461.bound LeftAuthority138504.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority138461.bound, LeftAuthority138504.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority138461.actual selector witness) * (LeftAuthority138504.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound138508

namespace LeftBound138516
def owner : Owner := ⟨.program ⟨257⟩, ⟨65735⟩⟩
def transferEvent : Nat := 138516
def frameStart : Nat := 138406
def rule : BoundRule := .sum [.predecessor 0 138514 .coefficient, .predecessor 1 138515 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 138514 .coefficient)
      LeftAuthority138512.bound (LeftAuthority138512.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events541.exact138513RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority138512.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority138512.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 138515 .coefficient)
      LeftBound138508.bound (LeftBound138508.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events541.exact138510RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound138508.bound, RecordedBoundRefines] <;> decide)
      (LeftBound138508.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority138512.bound, LeftBound138508.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority138512.bound, LeftBound138508.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority138512.actual selector witness, LeftBound138508.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound138516

namespace LeftBound138520
def owner : Owner := ⟨.program ⟨257⟩, ⟨69167⟩⟩
def transferEvent : Nat := 138520
def frameStart : Nat := 138406
def rule : BoundRule := .sum [.predecessor 0 138518 .coefficient, .predecessor 1 138519 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 138518 .coefficient)
      LeftBound138516.bound (LeftBound138516.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events541.exact138517RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound138516.bound, RecordedBoundRefines] <;> decide)
      (LeftBound138516.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 138519 .coefficient)
      LeftBound138497.bound (LeftBound138497.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events541.exact138502RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound138497.bound, RecordedBoundRefines] <;> decide)
      (LeftBound138497.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound138516.bound, LeftBound138497.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound138516.bound, LeftBound138497.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound138516.actual selector witness, LeftBound138497.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound138520

namespace LeftBound138533
def owner : Owner := ⟨.program ⟨257⟩, ⟨69165⟩⟩
def transferEvent : Nat := 138533
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 138531 .coefficient, .predecessor 1 138532 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 138531 .coefficient)
      LeftBound138354.bound (LeftBound138354.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events541.exact138530RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound138354.bound, RecordedBoundRefines] <;> decide)
      (LeftBound138354.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 138532 .coefficient)
      LeftBound138337.bound (LeftBound138337.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events540.exact138344RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound138337.bound, RecordedBoundRefines] <;> decide)
      (LeftBound138337.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound138354.bound, LeftBound138337.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound138354.bound, LeftBound138337.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound138354.actual selector witness, LeftBound138337.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound138533

namespace LeftBound138536
def owner : Owner := ⟨.program ⟨257⟩, ⟨69165⟩⟩
def transferEvent : Nat := 138536
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 138530 .summary, .result 138344 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 138530 .summary)
      LeftBound138356.bound (LeftBound138356.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨67703⟩⟩) (rawTerms := some (Proof.Events541.exact138530RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound138356.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 138344 .summary)
      LeftBound138339.bound (LeftBound138339.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69164⟩⟩) (rawTerms := some (Proof.Events540.exact138344RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound138339.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound138356.bound, LeftBound138339.bound]
def bound : CoeffClass := .finite ⟨2998054127048462696448, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound138356.bound, LeftBound138339.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound138356.actual selector witness, LeftBound138339.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound138536

namespace LeftBound138540
def owner : Owner := ⟨.program ⟨257⟩, ⟨69626⟩⟩
def transferEvent : Nat := 138540
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 138538 .coefficient) (.predecessor 1 138539 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 138538 .coefficient)
      LeftBound138533.bound (LeftBound138533.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events541.exact138537RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound138533.bound, RecordedBoundRefines] <;> decide)
      (LeftBound138533.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 138539 .coefficient)
      LeftAuthority138259.bound (LeftAuthority138259.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events540.exact138260RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority138259.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority138259.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound138533.bound LeftAuthority138259.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound138533.bound, LeftAuthority138259.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound138533.actual selector witness) * (LeftAuthority138259.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound138540

namespace LeftBound138541
def owner : Owner := ⟨.program ⟨257⟩, ⟨69626⟩⟩
def transferEvent : Nat := 138541
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨69624⟩⟩]⟩ [⟨.result 138260 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 138260 .coefficient)
      LeftAuthority138259.bound (LeftAuthority138259.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨69624⟩⟩) (rawTerms := some (Proof.Events540.exact138260RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority138259.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority138259.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority138259.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority138259.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority138259.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound138541

namespace LeftBound138542
def owner : Owner := ⟨.program ⟨257⟩, ⟨69626⟩⟩
def transferEvent : Nat := 138542
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 138537 .summary) (.transfer 138541) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 138537 .summary)
      LeftBound138536.bound (LeftBound138536.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69165⟩⟩) (rawTerms := some (Proof.Events541.exact138537RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound138536.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 138541)
      LeftBound138541.bound (LeftBound138541.actual selector witness) := by
  exact .transfer (LeftBound138541.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound138536.bound LeftBound138541.bound
def bound : CoeffClass := .finite ⟨32191361068277440720800338411520, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound138536.bound, LeftBound138541.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound138536.actual selector witness) * (LeftBound138541.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound138542

namespace LeftBound138553
def owner : Owner := ⟨.program ⟨257⟩, ⟨67939⟩⟩
def transferEvent : Nat := 138553
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 138551 .coefficient) (.value (.predecessor 1 138552 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 138551 .coefficient)
      LeftAuthority138549.bound (LeftAuthority138549.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events541.exact138550RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority138549.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority138549.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 138552 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority138549.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority138549.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority138549.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound138553

namespace LeftBound138557
def owner : Owner := ⟨.program ⟨257⟩, ⟨67940⟩⟩
def transferEvent : Nat := 138557
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 138555 .coefficient) (.predecessor 1 138556 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 138555 .coefficient)
      LeftBound134492.bound (LeftBound134492.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events525.exact134495RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound134492.bound, RecordedBoundRefines] <;> decide)
      (LeftBound134492.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 138556 .coefficient)
      LeftBound138553.bound (LeftBound138553.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events541.exact138554RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound138553.bound, RecordedBoundRefines] <;> decide)
      (LeftBound138553.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound134492.bound LeftBound138553.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound134492.bound, LeftBound138553.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound134492.actual selector witness) * (LeftBound138553.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound138557

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
