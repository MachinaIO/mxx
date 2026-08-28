import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1696
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1740

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound257393
def owner : Owner := ⟨.program ⟨257⟩, ⟨55248⟩⟩
def transferEvent : Nat := 257393
def frameStart : Nat := 257334
def rule : BoundRule := .product (.predecessor 0 257391 .coefficient) (.predecessor 1 257392 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 257391 .coefficient)
      LeftAuthority257389.bound (LeftAuthority257389.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1005.exact257390RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority257389.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority257389.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 257392 .coefficient)
      LeftBound257387.bound (LeftBound257387.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1005.exact257388RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound257387.bound, RecordedBoundRefines] <;> decide)
      (LeftBound257387.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority257389.bound LeftBound257387.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority257389.bound, LeftBound257387.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority257389.actual selector witness) * (LeftBound257387.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound257393

namespace LeftBound257409
def owner : Owner := ⟨.program ⟨257⟩, ⟨9530⟩⟩
def transferEvent : Nat := 257409
def frameStart : Nat := 257334
def rule : BoundRule := .scale (.predecessor 0 257407 .coefficient) (.value (.predecessor 1 257408 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 257407 .coefficient)
      LeftAuthority257405.bound (LeftAuthority257405.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1005.exact257406RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority257405.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority257405.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 257408 .coefficient)
      LeftAuthority257396.bound (LeftAuthority257396.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority257396.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority257405.bound LeftAuthority257396.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority257405.bound, LeftAuthority257396.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority257405.actual selector witness) * (LeftAuthority257396.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound257409

namespace LeftBound257412
def owner : Owner := ⟨.program ⟨257⟩, ⟨7289⟩⟩
def transferEvent : Nat := 257412
def frameStart : Nat := 257334
def rule : BoundRule := .identity (.predecessor 0 257411 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 257411 .coefficient)
      LeftAuthority257399.bound (LeftAuthority257399.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1005.exact257400RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority257399.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority257399.derived selector witness)

def rawBound : CoeffClass := LeftAuthority257399.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority257399.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority257399.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound257412

namespace LeftBound257416
def owner : Owner := ⟨.program ⟨257⟩, ⟨9531⟩⟩
def transferEvent : Nat := 257416
def frameStart : Nat := 257334
def rule : BoundRule := .product (.predecessor 0 257414 .coefficient) (.predecessor 1 257415 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 257414 .coefficient)
      LeftBound257412.bound (LeftBound257412.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1005.exact257413RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound257412.bound, RecordedBoundRefines] <;> decide)
      (LeftBound257412.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 257415 .coefficient)
      LeftBound257409.bound (LeftBound257409.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1005.exact257410RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound257409.bound, RecordedBoundRefines] <;> decide)
      (LeftBound257409.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound257412.bound LeftBound257409.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound257412.bound, LeftBound257409.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound257412.actual selector witness) * (LeftBound257409.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound257416

namespace LeftBound257421
def owner : Owner := ⟨.program ⟨257⟩, ⟨55249⟩⟩
def transferEvent : Nat := 257421
def frameStart : Nat := 257334
def rule : BoundRule := .sum [.predecessor 0 257419 .coefficient, .predecessor 1 257420 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 257419 .coefficient)
      LeftBound257416.bound (LeftBound257416.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1005.exact257418RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound257416.bound, RecordedBoundRefines] <;> decide)
      (LeftBound257416.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 257420 .coefficient)
      LeftBound257393.bound (LeftBound257393.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1005.exact257395RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound257393.bound, RecordedBoundRefines] <;> decide)
      (LeftBound257393.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound257416.bound, LeftBound257393.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound257416.bound, LeftBound257393.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound257416.actual selector witness, LeftBound257393.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound257421

namespace LeftBound257425
def owner : Owner := ⟨.program ⟨257⟩, ⟨55447⟩⟩
def transferEvent : Nat := 257425
def frameStart : Nat := 257334
def rule : BoundRule := .product (.predecessor 0 257423 .coefficient) (.predecessor 1 257424 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 257423 .coefficient)
      LeftBound257421.bound (LeftBound257421.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1005.exact257422RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound257421.bound, RecordedBoundRefines] <;> decide)
      (LeftBound257421.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 257424 .coefficient)
      LeftAuthority257378.bound (LeftAuthority257378.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1005.exact257379RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority257378.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority257378.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound257421.bound LeftAuthority257378.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound257421.bound, LeftAuthority257378.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound257421.actual selector witness) * (LeftAuthority257378.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound257425

namespace LeftBound257436
def owner : Owner := ⟨.program ⟨257⟩, ⟨53830⟩⟩
def transferEvent : Nat := 257436
def frameStart : Nat := 257334
def rule : BoundRule := .product (.predecessor 0 257434 .coefficient) (.predecessor 1 257435 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 257434 .coefficient)
      LeftAuthority257389.bound (LeftAuthority257389.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1005.exact257390RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority257389.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority257389.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 257435 .coefficient)
      LeftAuthority257432.bound (LeftAuthority257432.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1005.exact257433RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority257432.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority257432.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority257389.bound LeftAuthority257432.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority257389.bound, LeftAuthority257432.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority257389.actual selector witness) * (LeftAuthority257432.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound257436

namespace LeftBound257444
def owner : Owner := ⟨.program ⟨257⟩, ⟨53831⟩⟩
def transferEvent : Nat := 257444
def frameStart : Nat := 257334
def rule : BoundRule := .sum [.predecessor 0 257442 .coefficient, .predecessor 1 257443 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 257442 .coefficient)
      LeftAuthority257440.bound (LeftAuthority257440.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1005.exact257441RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority257440.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority257440.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 257443 .coefficient)
      LeftBound257436.bound (LeftBound257436.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1005.exact257438RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound257436.bound, RecordedBoundRefines] <;> decide)
      (LeftBound257436.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority257440.bound, LeftBound257436.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority257440.bound, LeftBound257436.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority257440.actual selector witness, LeftBound257436.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound257444

namespace LeftBound257448
def owner : Owner := ⟨.program ⟨257⟩, ⟨55448⟩⟩
def transferEvent : Nat := 257448
def frameStart : Nat := 257334
def rule : BoundRule := .sum [.predecessor 0 257446 .coefficient, .predecessor 1 257447 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 257446 .coefficient)
      LeftBound257444.bound (LeftBound257444.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1005.exact257445RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound257444.bound, RecordedBoundRefines] <;> decide)
      (LeftBound257444.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 257447 .coefficient)
      LeftBound257425.bound (LeftBound257425.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1005.exact257430RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound257425.bound, RecordedBoundRefines] <;> decide)
      (LeftBound257425.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound257444.bound, LeftBound257425.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound257444.bound, LeftBound257425.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound257444.actual selector witness, LeftBound257425.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound257448

namespace LeftBound257461
def owner : Owner := ⟨.program ⟨257⟩, ⟨55446⟩⟩
def transferEvent : Nat := 257461
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 257459 .coefficient, .predecessor 1 257460 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 257459 .coefficient)
      LeftBound257282.bound (LeftBound257282.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1005.exact257458RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound257282.bound, RecordedBoundRefines] <;> decide)
      (LeftBound257282.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 257460 .coefficient)
      LeftBound257265.bound (LeftBound257265.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1004.exact257272RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound257265.bound, RecordedBoundRefines] <;> decide)
      (LeftBound257265.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound257282.bound, LeftBound257265.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound257282.bound, LeftBound257265.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound257282.actual selector witness, LeftBound257265.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound257461

namespace LeftBound257464
def owner : Owner := ⟨.program ⟨257⟩, ⟨55446⟩⟩
def transferEvent : Nat := 257464
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 257458 .summary, .result 257272 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 257458 .summary)
      LeftBound257284.bound (LeftBound257284.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨54382⟩⟩) (rawTerms := some (Proof.Events1005.exact257458RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound257284.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 257272 .summary)
      LeftBound257267.bound (LeftBound257267.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55445⟩⟩) (rawTerms := some (Proof.Events1004.exact257272RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound257267.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound257284.bound, LeftBound257267.bound]
def bound : CoeffClass := .finite ⟨2997907760060573155328, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound257284.bound, LeftBound257267.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound257284.actual selector witness, LeftBound257267.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound257464

namespace LeftBound257468
def owner : Owner := ⟨.program ⟨257⟩, ⟨55779⟩⟩
def transferEvent : Nat := 257468
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 257466 .coefficient) (.predecessor 1 257467 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 257466 .coefficient)
      LeftBound257461.bound (LeftBound257461.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1005.exact257465RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound257461.bound, RecordedBoundRefines] <;> decide)
      (LeftBound257461.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 257467 .coefficient)
      LeftAuthority257187.bound (LeftAuthority257187.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1004.exact257188RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority257187.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority257187.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound257461.bound LeftAuthority257187.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound257461.bound, LeftAuthority257187.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound257461.actual selector witness) * (LeftAuthority257187.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound257468

namespace LeftBound257469
def owner : Owner := ⟨.program ⟨257⟩, ⟨55779⟩⟩
def transferEvent : Nat := 257469
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨55777⟩⟩]⟩ [⟨.result 257188 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 257188 .coefficient)
      LeftAuthority257187.bound (LeftAuthority257187.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨55777⟩⟩) (rawTerms := some (Proof.Events1004.exact257188RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority257187.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority257187.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority257187.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority257187.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority257187.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound257469

namespace LeftBound257470
def owner : Owner := ⟨.program ⟨257⟩, ⟨55779⟩⟩
def transferEvent : Nat := 257470
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 257465 .summary) (.transfer 257469) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 257465 .summary)
      LeftBound257464.bound (LeftBound257464.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55446⟩⟩) (rawTerms := some (Proof.Events1005.exact257465RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound257464.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 257469)
      LeftBound257469.bound (LeftBound257469.actual selector witness) := by
  exact .transfer (LeftBound257469.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound257464.bound LeftBound257469.bound
def bound : CoeffClass := .finite ⟨32189789464711941702873220382720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound257464.bound, LeftBound257469.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound257464.actual selector witness) * (LeftBound257469.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound257470

namespace LeftBound257481
def owner : Owner := ⟨.program ⟨257⟩, ⟨54638⟩⟩
def transferEvent : Nat := 257481
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 257479 .coefficient) (.value (.predecessor 1 257480 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 257479 .coefficient)
      LeftAuthority257477.bound (LeftAuthority257477.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1005.exact257478RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority257477.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority257477.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 257480 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority257477.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority257477.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority257477.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound257481

namespace LeftBound257485
def owner : Owner := ⟨.program ⟨257⟩, ⟨54639⟩⟩
def transferEvent : Nat := 257485
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 257483 .coefficient) (.predecessor 1 257484 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 257483 .coefficient)
      LeftBound251492.bound (LeftBound251492.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events982.exact251495RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound251492.bound, RecordedBoundRefines] <;> decide)
      (LeftBound251492.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 257484 .coefficient)
      LeftBound257481.bound (LeftBound257481.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1005.exact257482RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound257481.bound, RecordedBoundRefines] <;> decide)
      (LeftBound257481.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound251492.bound LeftBound257481.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound251492.bound, LeftBound257481.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound251492.actual selector witness) * (LeftBound257481.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound257485

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
