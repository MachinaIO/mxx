import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard682

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound105350
def owner : Owner := ⟨.program ⟨257⟩, ⟨49430⟩⟩
def transferEvent : Nat := 105350
def frameStart : Nat := 105300
def rule : BoundRule := .sum [.predecessor 0 105348 .coefficient, .predecessor 1 105349 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 105348 .coefficient)
      LeftBound105333.bound (LeftBound105333.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound105333.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 105349 .coefficient)
      LeftAuthority105346.bound (LeftAuthority105346.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority105346.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound105333.bound, LeftAuthority105346.bound]
def bound : CoeffClass := .finite ⟨3600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105333.bound, LeftAuthority105346.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound105333.actual selector witness, LeftAuthority105346.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound105350

namespace LeftBound105353
def owner : Owner := ⟨.program ⟨257⟩, ⟨49431⟩⟩
def transferEvent : Nat := 105353
def frameStart : Nat := 105300
def rule : BoundRule := .identity (.predecessor 0 105352 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 105352 .coefficient)
      LeftBound105350.bound (LeftBound105350.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound105350.derived selector witness)

def rawBound : CoeffClass := LeftBound105350.bound
def bound : CoeffClass := .finite ⟨3600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105350.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound105350.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound105353

namespace LeftBound105359
def owner : Owner := ⟨.program ⟨257⟩, ⟨49432⟩⟩
def transferEvent : Nat := 105359
def frameStart : Nat := 105300
def rule : BoundRule := .product (.predecessor 0 105357 .coefficient) (.predecessor 1 105358 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 105357 .coefficient)
      LeftAuthority105355.bound (LeftAuthority105355.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105356RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105355.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105355.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 105358 .coefficient)
      LeftBound105353.bound (LeftBound105353.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105354RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105353.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105353.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority105355.bound LeftBound105353.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105355.bound, LeftBound105353.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority105355.actual selector witness) * (LeftBound105353.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105359

namespace LeftBound105375
def owner : Owner := ⟨.program ⟨257⟩, ⟨9566⟩⟩
def transferEvent : Nat := 105375
def frameStart : Nat := 105300
def rule : BoundRule := .scale (.predecessor 0 105373 .coefficient) (.value (.predecessor 1 105374 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 105373 .coefficient)
      LeftAuthority105371.bound (LeftAuthority105371.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105372RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105371.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105371.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 105374 .coefficient)
      LeftAuthority105362.bound (LeftAuthority105362.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority105362.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority105371.bound LeftAuthority105362.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105371.bound, LeftAuthority105362.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority105371.actual selector witness) * (LeftAuthority105362.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound105375

namespace LeftBound105378
def owner : Owner := ⟨.program ⟨257⟩, ⟨7302⟩⟩
def transferEvent : Nat := 105378
def frameStart : Nat := 105300
def rule : BoundRule := .identity (.predecessor 0 105377 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 105377 .coefficient)
      LeftAuthority105365.bound (LeftAuthority105365.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105366RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105365.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105365.derived selector witness)

def rawBound : CoeffClass := LeftAuthority105365.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105365.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority105365.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound105378

namespace LeftBound105382
def owner : Owner := ⟨.program ⟨257⟩, ⟨9567⟩⟩
def transferEvent : Nat := 105382
def frameStart : Nat := 105300
def rule : BoundRule := .product (.predecessor 0 105380 .coefficient) (.predecessor 1 105381 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 105380 .coefficient)
      LeftBound105378.bound (LeftBound105378.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105379RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105378.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105378.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 105381 .coefficient)
      LeftBound105375.bound (LeftBound105375.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105376RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105375.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105375.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound105378.bound LeftBound105375.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105378.bound, LeftBound105375.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound105378.actual selector witness) * (LeftBound105375.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105382

namespace LeftBound105387
def owner : Owner := ⟨.program ⟨257⟩, ⟨49433⟩⟩
def transferEvent : Nat := 105387
def frameStart : Nat := 105300
def rule : BoundRule := .sum [.predecessor 0 105385 .coefficient, .predecessor 1 105386 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 105385 .coefficient)
      LeftBound105382.bound (LeftBound105382.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105384RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105382.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105382.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 105386 .coefficient)
      LeftBound105359.bound (LeftBound105359.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105361RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105359.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105359.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound105382.bound, LeftBound105359.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105382.bound, LeftBound105359.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound105382.actual selector witness, LeftBound105359.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound105387

namespace LeftBound105391
def owner : Owner := ⟨.program ⟨257⟩, ⟨49673⟩⟩
def transferEvent : Nat := 105391
def frameStart : Nat := 105300
def rule : BoundRule := .product (.predecessor 0 105389 .coefficient) (.predecessor 1 105390 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 105389 .coefficient)
      LeftBound105387.bound (LeftBound105387.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105388RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105387.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105387.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 105390 .coefficient)
      LeftAuthority105344.bound (LeftAuthority105344.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105345RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105344.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105344.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound105387.bound LeftAuthority105344.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105387.bound, LeftAuthority105344.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound105387.actual selector witness) * (LeftAuthority105344.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105391

namespace LeftBound105402
def owner : Owner := ⟨.program ⟨257⟩, ⟨48158⟩⟩
def transferEvent : Nat := 105402
def frameStart : Nat := 105300
def rule : BoundRule := .product (.predecessor 0 105400 .coefficient) (.predecessor 1 105401 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 105400 .coefficient)
      LeftAuthority105355.bound (LeftAuthority105355.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105356RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105355.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105355.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 105401 .coefficient)
      LeftAuthority105398.bound (LeftAuthority105398.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105399RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105398.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105398.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority105355.bound LeftAuthority105398.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105355.bound, LeftAuthority105398.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority105355.actual selector witness) * (LeftAuthority105398.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105402

namespace LeftBound105410
def owner : Owner := ⟨.program ⟨257⟩, ⟨48159⟩⟩
def transferEvent : Nat := 105410
def frameStart : Nat := 105300
def rule : BoundRule := .sum [.predecessor 0 105408 .coefficient, .predecessor 1 105409 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 105408 .coefficient)
      LeftAuthority105406.bound (LeftAuthority105406.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105407RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105406.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105406.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 105409 .coefficient)
      LeftBound105402.bound (LeftBound105402.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105404RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105402.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105402.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority105406.bound, LeftBound105402.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105406.bound, LeftBound105402.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority105406.actual selector witness, LeftBound105402.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound105410

namespace LeftBound105414
def owner : Owner := ⟨.program ⟨257⟩, ⟨49674⟩⟩
def transferEvent : Nat := 105414
def frameStart : Nat := 105300
def rule : BoundRule := .sum [.predecessor 0 105412 .coefficient, .predecessor 1 105413 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 105412 .coefficient)
      LeftBound105410.bound (LeftBound105410.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105411RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105410.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105410.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 105413 .coefficient)
      LeftBound105391.bound (LeftBound105391.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105396RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105391.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105391.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound105410.bound, LeftBound105391.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105410.bound, LeftBound105391.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound105410.actual selector witness, LeftBound105391.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound105414

namespace LeftBound105427
def owner : Owner := ⟨.program ⟨257⟩, ⟨49672⟩⟩
def transferEvent : Nat := 105427
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 105425 .coefficient, .predecessor 1 105426 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 105425 .coefficient)
      LeftBound105248.bound (LeftBound105248.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105424RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105248.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105248.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 105426 .coefficient)
      LeftBound105220.bound (LeftBound105220.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105227RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105220.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105220.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound105248.bound, LeftBound105220.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105248.bound, LeftBound105220.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound105248.actual selector witness, LeftBound105220.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound105427

namespace LeftBound105430
def owner : Owner := ⟨.program ⟨257⟩, ⟨49672⟩⟩
def transferEvent : Nat := 105430
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 105424 .summary, .result 105227 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 105424 .summary)
      LeftBound105250.bound (LeftBound105250.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨48602⟩⟩) (rawTerms := some (Proof.Events411.exact105424RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound105250.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 105227 .summary)
      LeftBound105222.bound (LeftBound105222.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨49671⟩⟩) (rawTerms := some (Proof.Events411.exact105227RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound105222.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound105250.bound, LeftBound105222.bound]
def bound : CoeffClass := .finite ⟨2998346861024241778688, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105250.bound, LeftBound105222.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound105250.actual selector witness, LeftBound105222.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound105430

namespace LeftBound105434
def owner : Owner := ⟨.program ⟨257⟩, ⟨50056⟩⟩
def transferEvent : Nat := 105434
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 105432 .coefficient) (.predecessor 1 105433 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 105432 .coefficient)
      LeftBound105427.bound (LeftBound105427.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105431RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105427.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105427.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 105433 .coefficient)
      LeftAuthority105137.bound (LeftAuthority105137.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events410.exact105138RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105137.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105137.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound105427.bound LeftAuthority105137.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105427.bound, LeftAuthority105137.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound105427.actual selector witness) * (LeftAuthority105137.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105434

namespace LeftBound105435
def owner : Owner := ⟨.program ⟨257⟩, ⟨50056⟩⟩
def transferEvent : Nat := 105435
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨50054⟩⟩]⟩ [⟨.result 105138 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 105138 .coefficient)
      LeftAuthority105137.bound (LeftAuthority105137.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨50054⟩⟩) (rawTerms := some (Proof.Events410.exact105138RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105137.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105137.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority105137.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105137.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority105137.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound105435

namespace LeftBound105436
def owner : Owner := ⟨.program ⟨257⟩, ⟨50056⟩⟩
def transferEvent : Nat := 105436
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 105431 .summary) (.transfer 105435) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 105431 .summary)
      LeftBound105430.bound (LeftBound105430.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨49672⟩⟩) (rawTerms := some (Proof.Events411.exact105431RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound105430.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 105435)
      LeftBound105435.bound (LeftBound105435.actual selector witness) := by
  exact .transfer (LeftBound105435.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound105430.bound LeftBound105435.bound
def bound : CoeffClass := .finite ⟨32194504275408438756654574469120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105430.bound, LeftBound105435.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound105430.actual selector witness) * (LeftBound105435.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105436

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
