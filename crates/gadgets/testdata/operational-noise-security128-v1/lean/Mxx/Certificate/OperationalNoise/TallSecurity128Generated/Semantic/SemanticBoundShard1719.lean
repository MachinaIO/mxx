import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1718

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound254471
def owner : Owner := ⟨.program ⟨257⟩, ⟨28655⟩⟩
def transferEvent : Nat := 254471
def frameStart : Nat := 254442
def rule : BoundRule := .product (.predecessor 0 254469 .coefficient) (.predecessor 1 254470 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 254469 .coefficient)
      LeftAuthority254467.bound (LeftAuthority254467.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events994.exact254468RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority254467.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority254467.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 254470 .coefficient)
      LeftAuthority254464.bound (LeftAuthority254464.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events994.exact254465RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority254464.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority254464.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority254467.bound LeftAuthority254464.bound
def bound : CoeffClass := .finite ⟨1296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority254467.bound, LeftAuthority254464.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority254467.actual selector witness) * (LeftAuthority254464.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound254471

namespace LeftBound254475
def owner : Owner := ⟨.program ⟨257⟩, ⟨28656⟩⟩
def transferEvent : Nat := 254475
def frameStart : Nat := 254442
def rule : BoundRule := .identity (.predecessor 0 254474 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 254474 .coefficient)
      LeftBound254471.bound (LeftBound254471.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events994.exact254473RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound254471.bound, RecordedBoundRefines] <;> decide)
      (LeftBound254471.derived selector witness)

def rawBound : CoeffClass := LeftBound254471.bound
def bound : CoeffClass := .finite ⟨1296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound254471.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound254471.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound254475

namespace LeftBound254492
def owner : Owner := ⟨.program ⟨257⟩, ⟨30346⟩⟩
def transferEvent : Nat := 254492
def frameStart : Nat := 254442
def rule : BoundRule := .sum [.predecessor 0 254490 .coefficient, .predecessor 1 254491 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 254490 .coefficient)
      LeftBound254475.bound (LeftBound254475.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound254475.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 254491 .coefficient)
      LeftAuthority254488.bound (LeftAuthority254488.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority254488.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound254475.bound, LeftAuthority254488.bound]
def bound : CoeffClass := .finite ⟨1296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound254475.bound, LeftAuthority254488.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound254475.actual selector witness, LeftAuthority254488.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound254492

namespace LeftBound254495
def owner : Owner := ⟨.program ⟨257⟩, ⟨30347⟩⟩
def transferEvent : Nat := 254495
def frameStart : Nat := 254442
def rule : BoundRule := .identity (.predecessor 0 254494 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 254494 .coefficient)
      LeftBound254492.bound (LeftBound254492.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound254492.derived selector witness)

def rawBound : CoeffClass := LeftBound254492.bound
def bound : CoeffClass := .finite ⟨1296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound254492.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound254492.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound254495

namespace LeftBound254501
def owner : Owner := ⟨.program ⟨257⟩, ⟨30348⟩⟩
def transferEvent : Nat := 254501
def frameStart : Nat := 254442
def rule : BoundRule := .product (.predecessor 0 254499 .coefficient) (.predecessor 1 254500 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 254499 .coefficient)
      LeftAuthority254497.bound (LeftAuthority254497.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events994.exact254498RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority254497.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority254497.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 254500 .coefficient)
      LeftBound254495.bound (LeftBound254495.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events994.exact254496RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound254495.bound, RecordedBoundRefines] <;> decide)
      (LeftBound254495.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority254497.bound LeftBound254495.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority254497.bound, LeftBound254495.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority254497.actual selector witness) * (LeftBound254495.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound254501

namespace LeftBound254517
def owner : Owner := ⟨.program ⟨257⟩, ⟨9548⟩⟩
def transferEvent : Nat := 254517
def frameStart : Nat := 254442
def rule : BoundRule := .scale (.predecessor 0 254515 .coefficient) (.value (.predecessor 1 254516 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 254515 .coefficient)
      LeftAuthority254513.bound (LeftAuthority254513.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events994.exact254514RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority254513.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority254513.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 254516 .coefficient)
      LeftAuthority254504.bound (LeftAuthority254504.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority254504.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority254513.bound LeftAuthority254504.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority254513.bound, LeftAuthority254504.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority254513.actual selector witness) * (LeftAuthority254504.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound254517

namespace LeftBound254520
def owner : Owner := ⟨.program ⟨257⟩, ⟨7296⟩⟩
def transferEvent : Nat := 254520
def frameStart : Nat := 254442
def rule : BoundRule := .identity (.predecessor 0 254519 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 254519 .coefficient)
      LeftAuthority254507.bound (LeftAuthority254507.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events994.exact254508RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority254507.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority254507.derived selector witness)

def rawBound : CoeffClass := LeftAuthority254507.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority254507.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority254507.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound254520

namespace LeftBound254524
def owner : Owner := ⟨.program ⟨257⟩, ⟨9549⟩⟩
def transferEvent : Nat := 254524
def frameStart : Nat := 254442
def rule : BoundRule := .product (.predecessor 0 254522 .coefficient) (.predecessor 1 254523 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 254522 .coefficient)
      LeftBound254520.bound (LeftBound254520.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events994.exact254521RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound254520.bound, RecordedBoundRefines] <;> decide)
      (LeftBound254520.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 254523 .coefficient)
      LeftBound254517.bound (LeftBound254517.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events994.exact254518RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound254517.bound, RecordedBoundRefines] <;> decide)
      (LeftBound254517.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound254520.bound LeftBound254517.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound254520.bound, LeftBound254517.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound254520.actual selector witness) * (LeftBound254517.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound254524

namespace LeftBound254529
def owner : Owner := ⟨.program ⟨257⟩, ⟨30349⟩⟩
def transferEvent : Nat := 254529
def frameStart : Nat := 254442
def rule : BoundRule := .sum [.predecessor 0 254527 .coefficient, .predecessor 1 254528 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 254527 .coefficient)
      LeftBound254524.bound (LeftBound254524.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events994.exact254526RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound254524.bound, RecordedBoundRefines] <;> decide)
      (LeftBound254524.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 254528 .coefficient)
      LeftBound254501.bound (LeftBound254501.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events994.exact254503RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound254501.bound, RecordedBoundRefines] <;> decide)
      (LeftBound254501.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound254524.bound, LeftBound254501.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound254524.bound, LeftBound254501.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound254524.actual selector witness, LeftBound254501.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound254529

namespace LeftBound254533
def owner : Owner := ⟨.program ⟨257⟩, ⟨30547⟩⟩
def transferEvent : Nat := 254533
def frameStart : Nat := 254442
def rule : BoundRule := .product (.predecessor 0 254531 .coefficient) (.predecessor 1 254532 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 254531 .coefficient)
      LeftBound254529.bound (LeftBound254529.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events994.exact254530RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound254529.bound, RecordedBoundRefines] <;> decide)
      (LeftBound254529.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 254532 .coefficient)
      LeftAuthority254486.bound (LeftAuthority254486.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events994.exact254487RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority254486.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority254486.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound254529.bound LeftAuthority254486.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound254529.bound, LeftAuthority254486.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound254529.actual selector witness) * (LeftAuthority254486.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound254533

namespace LeftBound254544
def owner : Owner := ⟨.program ⟨257⟩, ⟨29050⟩⟩
def transferEvent : Nat := 254544
def frameStart : Nat := 254442
def rule : BoundRule := .product (.predecessor 0 254542 .coefficient) (.predecessor 1 254543 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 254542 .coefficient)
      LeftAuthority254497.bound (LeftAuthority254497.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events994.exact254498RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority254497.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority254497.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 254543 .coefficient)
      LeftAuthority254540.bound (LeftAuthority254540.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events994.exact254541RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority254540.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority254540.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority254497.bound LeftAuthority254540.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority254497.bound, LeftAuthority254540.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority254497.actual selector witness) * (LeftAuthority254540.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound254544

namespace LeftBound254552
def owner : Owner := ⟨.program ⟨257⟩, ⟨29051⟩⟩
def transferEvent : Nat := 254552
def frameStart : Nat := 254442
def rule : BoundRule := .sum [.predecessor 0 254550 .coefficient, .predecessor 1 254551 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 254550 .coefficient)
      LeftAuthority254548.bound (LeftAuthority254548.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events994.exact254549RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority254548.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority254548.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 254551 .coefficient)
      LeftBound254544.bound (LeftBound254544.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events994.exact254546RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound254544.bound, RecordedBoundRefines] <;> decide)
      (LeftBound254544.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority254548.bound, LeftBound254544.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority254548.bound, LeftBound254544.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority254548.actual selector witness, LeftBound254544.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound254552

namespace LeftBound254556
def owner : Owner := ⟨.program ⟨257⟩, ⟨30548⟩⟩
def transferEvent : Nat := 254556
def frameStart : Nat := 254442
def rule : BoundRule := .sum [.predecessor 0 254554 .coefficient, .predecessor 1 254555 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 254554 .coefficient)
      LeftBound254552.bound (LeftBound254552.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events994.exact254553RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound254552.bound, RecordedBoundRefines] <;> decide)
      (LeftBound254552.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 254555 .coefficient)
      LeftBound254533.bound (LeftBound254533.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events994.exact254538RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound254533.bound, RecordedBoundRefines] <;> decide)
      (LeftBound254533.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound254552.bound, LeftBound254533.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound254552.bound, LeftBound254533.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound254552.actual selector witness, LeftBound254533.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound254556

namespace LeftBound254569
def owner : Owner := ⟨.program ⟨257⟩, ⟨30546⟩⟩
def transferEvent : Nat := 254569
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 254567 .coefficient, .predecessor 1 254568 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 254567 .coefficient)
      LeftBound254390.bound (LeftBound254390.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events994.exact254566RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound254390.bound, RecordedBoundRefines] <;> decide)
      (LeftBound254390.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 254568 .coefficient)
      LeftBound254373.bound (LeftBound254373.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events993.exact254380RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound254373.bound, RecordedBoundRefines] <;> decide)
      (LeftBound254373.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound254390.bound, LeftBound254373.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound254390.bound, LeftBound254373.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound254390.actual selector witness, LeftBound254373.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound254569

namespace LeftBound254572
def owner : Owner := ⟨.program ⟨257⟩, ⟨30546⟩⟩
def transferEvent : Nat := 254572
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 254566 .summary, .result 254380 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 254566 .summary)
      LeftBound254392.bound (LeftBound254392.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨29482⟩⟩) (rawTerms := some (Proof.Events994.exact254566RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound254392.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 254380 .summary)
      LeftBound254375.bound (LeftBound254375.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨30545⟩⟩) (rawTerms := some (Proof.Events993.exact254380RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound254375.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound254392.bound, LeftBound254375.bound]
def bound : CoeffClass := .finite ⟨2998127310542407467008, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound254392.bound, LeftBound254375.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound254392.actual selector witness, LeftBound254375.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound254572

namespace LeftBound254576
def owner : Owner := ⟨.program ⟨257⟩, ⟨30846⟩⟩
def transferEvent : Nat := 254576
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 254574 .coefficient) (.predecessor 1 254575 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 254574 .coefficient)
      LeftBound254569.bound (LeftBound254569.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events994.exact254573RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound254569.bound, RecordedBoundRefines] <;> decide)
      (LeftBound254569.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 254575 .coefficient)
      LeftAuthority254295.bound (LeftAuthority254295.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events993.exact254296RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority254295.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority254295.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound254569.bound LeftAuthority254295.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound254569.bound, LeftAuthority254295.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound254569.actual selector witness) * (LeftAuthority254295.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound254576

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
