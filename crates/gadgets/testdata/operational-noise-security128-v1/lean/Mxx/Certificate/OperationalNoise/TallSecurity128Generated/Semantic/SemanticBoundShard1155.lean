import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1154

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound173531
def owner : Owner := ⟨.program ⟨257⟩, ⟨32183⟩⟩
def transferEvent : Nat := 173531
def frameStart : Nat := 173086
def rule : BoundRule := .sum [.predecessor 0 173529 .coefficient, .predecessor 1 173530 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 173529 .coefficient)
      LeftBound173527.bound (LeftBound173527.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events677.exact173528RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound173527.bound, RecordedBoundRefines] <;> decide)
      (LeftBound173527.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 173530 .coefficient)
      LeftAuthority173450.bound (LeftAuthority173450.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events677.exact173451RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority173450.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority173450.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound173527.bound, LeftAuthority173450.bound]
def bound : CoeffClass := .finite ⟨197, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound173527.bound, LeftAuthority173450.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound173527.actual selector witness, LeftAuthority173450.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound173531

namespace LeftBound173535
def owner : Owner := ⟨.program ⟨257⟩, ⟨51238⟩⟩
def transferEvent : Nat := 173535
def frameStart : Nat := 173086
def rule : BoundRule := .sum [.predecessor 0 173533 .coefficient, .predecessor 1 173534 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 173533 .coefficient)
      LeftBound173531.bound (LeftBound173531.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events677.exact173532RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound173531.bound, RecordedBoundRefines] <;> decide)
      (LeftBound173531.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 173534 .coefficient)
      LeftAuthority173427.bound (LeftAuthority173427.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events677.exact173428RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority173427.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority173427.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound173531.bound, LeftAuthority173427.bound]
def bound : CoeffClass := .finite ⟨255, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound173531.bound, LeftAuthority173427.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound173531.actual selector witness, LeftAuthority173427.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound173535

namespace LeftBound173539
def owner : Owner := ⟨.program ⟨257⟩, ⟨54218⟩⟩
def transferEvent : Nat := 173539
def frameStart : Nat := 173086
def rule : BoundRule := .sum [.predecessor 0 173537 .coefficient, .predecessor 1 173538 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 173537 .coefficient)
      LeftBound173535.bound (LeftBound173535.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events677.exact173536RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound173535.bound, RecordedBoundRefines] <;> decide)
      (LeftBound173535.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 173538 .coefficient)
      LeftAuthority173404.bound (LeftAuthority173404.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events677.exact173405RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority173404.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority173404.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound173535.bound, LeftAuthority173404.bound]
def bound : CoeffClass := .finite ⟨314, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound173535.bound, LeftAuthority173404.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound173535.actual selector witness, LeftAuthority173404.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound173539

namespace LeftBound173543
def owner : Owner := ⟨.program ⟨257⟩, ⟨57198⟩⟩
def transferEvent : Nat := 173543
def frameStart : Nat := 173086
def rule : BoundRule := .sum [.predecessor 0 173541 .coefficient, .predecessor 1 173542 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 173541 .coefficient)
      LeftBound173539.bound (LeftBound173539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events677.exact173540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound173539.bound, RecordedBoundRefines] <;> decide)
      (LeftBound173539.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 173542 .coefficient)
      LeftAuthority173381.bound (LeftAuthority173381.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events677.exact173382RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority173381.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority173381.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound173539.bound, LeftAuthority173381.bound]
def bound : CoeffClass := .finite ⟨374, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound173539.bound, LeftAuthority173381.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound173539.actual selector witness, LeftAuthority173381.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound173543

namespace LeftBound173547
def owner : Owner := ⟨.program ⟨257⟩, ⟨60178⟩⟩
def transferEvent : Nat := 173547
def frameStart : Nat := 173086
def rule : BoundRule := .sum [.predecessor 0 173545 .coefficient, .predecessor 1 173546 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 173545 .coefficient)
      LeftBound173543.bound (LeftBound173543.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events677.exact173544RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound173543.bound, RecordedBoundRefines] <;> decide)
      (LeftBound173543.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 173546 .coefficient)
      LeftAuthority173358.bound (LeftAuthority173358.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events677.exact173359RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority173358.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority173358.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound173543.bound, LeftAuthority173358.bound]
def bound : CoeffClass := .finite ⟨435, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound173543.bound, LeftAuthority173358.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound173543.actual selector witness, LeftAuthority173358.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound173547

namespace LeftBound173551
def owner : Owner := ⟨.program ⟨257⟩, ⟨63158⟩⟩
def transferEvent : Nat := 173551
def frameStart : Nat := 173086
def rule : BoundRule := .sum [.predecessor 0 173549 .coefficient, .predecessor 1 173550 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 173549 .coefficient)
      LeftBound173547.bound (LeftBound173547.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events677.exact173548RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound173547.bound, RecordedBoundRefines] <;> decide)
      (LeftBound173547.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 173550 .coefficient)
      LeftAuthority173335.bound (LeftAuthority173335.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events677.exact173336RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority173335.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority173335.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound173547.bound, LeftAuthority173335.bound]
def bound : CoeffClass := .finite ⟨496, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound173547.bound, LeftAuthority173335.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound173547.actual selector witness, LeftAuthority173335.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound173551

namespace LeftBound173555
def owner : Owner := ⟨.program ⟨257⟩, ⟨66882⟩⟩
def transferEvent : Nat := 173555
def frameStart : Nat := 173086
def rule : BoundRule := .sum [.predecessor 0 173553 .coefficient, .predecessor 1 173554 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 173553 .coefficient)
      LeftBound173551.bound (LeftBound173551.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events677.exact173552RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound173551.bound, RecordedBoundRefines] <;> decide)
      (LeftBound173551.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 173554 .coefficient)
      LeftAuthority173312.bound (LeftAuthority173312.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events677.exact173313RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority173312.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority173312.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound173551.bound, LeftAuthority173312.bound]
def bound : CoeffClass := .finite ⟨558, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound173551.bound, LeftAuthority173312.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound173551.actual selector witness, LeftAuthority173312.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound173555

namespace LeftBound173559
def owner : Owner := ⟨.program ⟨257⟩, ⟨66883⟩⟩
def transferEvent : Nat := 173559
def frameStart : Nat := 173086
def rule : BoundRule := .sum [.predecessor 0 173557 .coefficient, .predecessor 1 173558 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 173557 .coefficient)
      LeftBound173555.bound (LeftBound173555.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events677.exact173556RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound173555.bound, RecordedBoundRefines] <;> decide)
      (LeftBound173555.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 173558 .coefficient)
      LeftAuthority173289.bound (LeftAuthority173289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events676.exact173290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority173289.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority173289.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound173555.bound, LeftAuthority173289.bound]
def bound : CoeffClass := .finite ⟨620, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound173555.bound, LeftAuthority173289.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound173555.actual selector witness, LeftAuthority173289.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound173559

namespace LeftBound173563
def owner : Owner := ⟨.program ⟨257⟩, ⟨66884⟩⟩
def transferEvent : Nat := 173563
def frameStart : Nat := 173086
def rule : BoundRule := .sum [.predecessor 0 173561 .coefficient, .predecessor 1 173562 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 173561 .coefficient)
      LeftBound173559.bound (LeftBound173559.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events677.exact173560RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound173559.bound, RecordedBoundRefines] <;> decide)
      (LeftBound173559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 173562 .coefficient)
      LeftAuthority173266.bound (LeftAuthority173266.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events676.exact173267RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority173266.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority173266.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound173559.bound, LeftAuthority173266.bound]
def bound : CoeffClass := .finite ⟨682, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound173559.bound, LeftAuthority173266.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound173559.actual selector witness, LeftAuthority173266.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound173563

namespace LeftBound173567
def owner : Owner := ⟨.program ⟨257⟩, ⟨66885⟩⟩
def transferEvent : Nat := 173567
def frameStart : Nat := 173086
def rule : BoundRule := .sum [.predecessor 0 173565 .coefficient, .predecessor 1 173566 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 173565 .coefficient)
      LeftBound173563.bound (LeftBound173563.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events677.exact173564RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound173563.bound, RecordedBoundRefines] <;> decide)
      (LeftBound173563.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 173566 .coefficient)
      LeftAuthority173243.bound (LeftAuthority173243.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events676.exact173244RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority173243.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority173243.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound173563.bound, LeftAuthority173243.bound]
def bound : CoeffClass := .finite ⟨744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound173563.bound, LeftAuthority173243.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound173563.actual selector witness, LeftAuthority173243.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound173567

namespace LeftBound173571
def owner : Owner := ⟨.program ⟨257⟩, ⟨66886⟩⟩
def transferEvent : Nat := 173571
def frameStart : Nat := 173086
def rule : BoundRule := .sum [.predecessor 0 173569 .coefficient, .predecessor 1 173570 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 173569 .coefficient)
      LeftBound173567.bound (LeftBound173567.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events678.exact173568RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound173567.bound, RecordedBoundRefines] <;> decide)
      (LeftBound173567.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 173570 .coefficient)
      LeftAuthority173220.bound (LeftAuthority173220.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events676.exact173221RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority173220.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority173220.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound173567.bound, LeftAuthority173220.bound]
def bound : CoeffClass := .finite ⟨807, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound173567.bound, LeftAuthority173220.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound173567.actual selector witness, LeftAuthority173220.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound173571

namespace LeftBound173575
def owner : Owner := ⟨.program ⟨257⟩, ⟨66887⟩⟩
def transferEvent : Nat := 173575
def frameStart : Nat := 173086
def rule : BoundRule := .sum [.predecessor 0 173573 .coefficient, .predecessor 1 173574 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 173573 .coefficient)
      LeftBound173571.bound (LeftBound173571.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events678.exact173572RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound173571.bound, RecordedBoundRefines] <;> decide)
      (LeftBound173571.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 173574 .coefficient)
      LeftAuthority173197.bound (LeftAuthority173197.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events676.exact173198RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority173197.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority173197.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound173571.bound, LeftAuthority173197.bound]
def bound : CoeffClass := .finite ⟨870, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound173571.bound, LeftAuthority173197.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound173571.actual selector witness, LeftAuthority173197.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound173575

namespace LeftBound173579
def owner : Owner := ⟨.program ⟨257⟩, ⟨66888⟩⟩
def transferEvent : Nat := 173579
def frameStart : Nat := 173086
def rule : BoundRule := .sum [.predecessor 0 173577 .coefficient, .predecessor 1 173578 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 173577 .coefficient)
      LeftBound173575.bound (LeftBound173575.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events678.exact173576RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound173575.bound, RecordedBoundRefines] <;> decide)
      (LeftBound173575.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 173578 .coefficient)
      LeftAuthority173174.bound (LeftAuthority173174.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events676.exact173175RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority173174.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority173174.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound173575.bound, LeftAuthority173174.bound]
def bound : CoeffClass := .finite ⟨933, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound173575.bound, LeftAuthority173174.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound173575.actual selector witness, LeftAuthority173174.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound173579

namespace LeftBound173583
def owner : Owner := ⟨.program ⟨257⟩, ⟨66889⟩⟩
def transferEvent : Nat := 173583
def frameStart : Nat := 173086
def rule : BoundRule := .sum [.predecessor 0 173581 .coefficient, .predecessor 1 173582 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 173581 .coefficient)
      LeftBound173579.bound (LeftBound173579.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events678.exact173580RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound173579.bound, RecordedBoundRefines] <;> decide)
      (LeftBound173579.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 173582 .coefficient)
      LeftAuthority173151.bound (LeftAuthority173151.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events676.exact173152RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority173151.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority173151.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound173579.bound, LeftAuthority173151.bound]
def bound : CoeffClass := .finite ⟨996, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound173579.bound, LeftAuthority173151.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound173579.actual selector witness, LeftAuthority173151.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound173583

namespace LeftBound173587
def owner : Owner := ⟨.program ⟨257⟩, ⟨66890⟩⟩
def transferEvent : Nat := 173587
def frameStart : Nat := 173086
def rule : BoundRule := .sum [.predecessor 0 173585 .coefficient, .predecessor 1 173586 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 173585 .coefficient)
      LeftBound173583.bound (LeftBound173583.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events678.exact173584RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound173583.bound, RecordedBoundRefines] <;> decide)
      (LeftBound173583.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 173586 .coefficient)
      LeftAuthority173128.bound (LeftAuthority173128.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events676.exact173129RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority173128.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority173128.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound173583.bound, LeftAuthority173128.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound173583.bound, LeftAuthority173128.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound173583.actual selector witness, LeftAuthority173128.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound173587

namespace LeftBound173590
def owner : Owner := ⟨.program ⟨257⟩, ⟨66891⟩⟩
def transferEvent : Nat := 173590
def frameStart : Nat := 173086
def rule : BoundRule := .identity (.predecessor 0 173589 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 173589 .coefficient)
      LeftBound173587.bound (LeftBound173587.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events678.exact173588RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound173587.bound, RecordedBoundRefines] <;> decide)
      (LeftBound173587.derived selector witness)

def rawBound : CoeffClass := LeftBound173587.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound173587.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound173587.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound173590

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
