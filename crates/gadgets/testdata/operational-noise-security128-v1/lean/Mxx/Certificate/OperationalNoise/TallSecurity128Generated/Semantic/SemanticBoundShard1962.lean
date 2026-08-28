import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1899
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1961

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound289019
def owner : Owner := ⟨.program ⟨257⟩, ⟨17104⟩⟩
def transferEvent : Nat := 289019
def frameStart : Nat := 288960
def rule : BoundRule := .product (.predecessor 0 289017 .coefficient) (.predecessor 1 289018 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 289017 .coefficient)
      LeftAuthority289015.bound (LeftAuthority289015.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1128.exact289016RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority289015.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority289015.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 289018 .coefficient)
      LeftBound289013.bound (LeftBound289013.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1128.exact289014RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound289013.bound, RecordedBoundRefines] <;> decide)
      (LeftBound289013.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority289015.bound LeftBound289013.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority289015.bound, LeftBound289013.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority289015.actual selector witness) * (LeftBound289013.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound289019

namespace LeftBound289033
def owner : Owner := ⟨.program ⟨257⟩, ⟨9569⟩⟩
def transferEvent : Nat := 289033
def frameStart : Nat := 288960
def rule : BoundRule := .scale (.predecessor 0 289031 .coefficient) (.value (.predecessor 1 289032 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 289031 .coefficient)
      LeftAuthority289029.bound (LeftAuthority289029.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1129.exact289030RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority289029.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority289029.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 289032 .coefficient)
      LeftAuthority288963.bound (LeftAuthority288963.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority288963.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority289029.bound LeftAuthority288963.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority289029.bound, LeftAuthority288963.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority289029.actual selector witness) * (LeftAuthority288963.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound289033

namespace LeftBound289036
def owner : Owner := ⟨.program ⟨257⟩, ⟨7303⟩⟩
def transferEvent : Nat := 289036
def frameStart : Nat := 288960
def rule : BoundRule := .identity (.predecessor 0 289035 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 289035 .coefficient)
      LeftAuthority289023.bound (LeftAuthority289023.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1129.exact289024RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority289023.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority289023.derived selector witness)

def rawBound : CoeffClass := LeftAuthority289023.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority289023.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority289023.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound289036

namespace LeftBound289040
def owner : Owner := ⟨.program ⟨257⟩, ⟨9570⟩⟩
def transferEvent : Nat := 289040
def frameStart : Nat := 288960
def rule : BoundRule := .product (.predecessor 0 289038 .coefficient) (.predecessor 1 289039 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 289038 .coefficient)
      LeftBound289036.bound (LeftBound289036.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1129.exact289037RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound289036.bound, RecordedBoundRefines] <;> decide)
      (LeftBound289036.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 289039 .coefficient)
      LeftBound289033.bound (LeftBound289033.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1129.exact289034RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound289033.bound, RecordedBoundRefines] <;> decide)
      (LeftBound289033.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound289036.bound LeftBound289033.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289036.bound, LeftBound289033.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound289036.actual selector witness) * (LeftBound289033.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound289040

namespace LeftBound289045
def owner : Owner := ⟨.program ⟨257⟩, ⟨17105⟩⟩
def transferEvent : Nat := 289045
def frameStart : Nat := 288960
def rule : BoundRule := .sum [.predecessor 0 289043 .coefficient, .predecessor 1 289044 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 289043 .coefficient)
      LeftBound289040.bound (LeftBound289040.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1129.exact289042RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound289040.bound, RecordedBoundRefines] <;> decide)
      (LeftBound289040.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 289044 .coefficient)
      LeftBound289019.bound (LeftBound289019.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1128.exact289021RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound289019.bound, RecordedBoundRefines] <;> decide)
      (LeftBound289019.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound289040.bound, LeftBound289019.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289040.bound, LeftBound289019.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound289040.actual selector witness, LeftBound289019.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289045

namespace LeftBound289049
def owner : Owner := ⟨.program ⟨257⟩, ⟨17296⟩⟩
def transferEvent : Nat := 289049
def frameStart : Nat := 288960
def rule : BoundRule := .product (.predecessor 0 289047 .coefficient) (.predecessor 1 289048 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 289047 .coefficient)
      LeftBound289045.bound (LeftBound289045.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1129.exact289046RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound289045.bound, RecordedBoundRefines] <;> decide)
      (LeftBound289045.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 289048 .coefficient)
      LeftAuthority289004.bound (LeftAuthority289004.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1128.exact289005RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority289004.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority289004.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound289045.bound LeftAuthority289004.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289045.bound, LeftAuthority289004.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound289045.actual selector witness) * (LeftAuthority289004.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound289049

namespace LeftBound289060
def owner : Owner := ⟨.program ⟨257⟩, ⟨15742⟩⟩
def transferEvent : Nat := 289060
def frameStart : Nat := 288960
def rule : BoundRule := .product (.predecessor 0 289058 .coefficient) (.predecessor 1 289059 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 289058 .coefficient)
      LeftAuthority289015.bound (LeftAuthority289015.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1128.exact289016RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority289015.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority289015.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 289059 .coefficient)
      LeftAuthority289056.bound (LeftAuthority289056.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1129.exact289057RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority289056.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority289056.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority289015.bound LeftAuthority289056.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority289015.bound, LeftAuthority289056.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority289015.actual selector witness) * (LeftAuthority289056.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound289060

namespace LeftBound289068
def owner : Owner := ⟨.program ⟨257⟩, ⟨15743⟩⟩
def transferEvent : Nat := 289068
def frameStart : Nat := 288960
def rule : BoundRule := .sum [.predecessor 0 289066 .coefficient, .predecessor 1 289067 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 289066 .coefficient)
      LeftAuthority289064.bound (LeftAuthority289064.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1129.exact289065RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority289064.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority289064.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 289067 .coefficient)
      LeftBound289060.bound (LeftBound289060.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1129.exact289062RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound289060.bound, RecordedBoundRefines] <;> decide)
      (LeftBound289060.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority289064.bound, LeftBound289060.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority289064.bound, LeftBound289060.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority289064.actual selector witness, LeftBound289060.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289068

namespace LeftBound289072
def owner : Owner := ⟨.program ⟨257⟩, ⟨17297⟩⟩
def transferEvent : Nat := 289072
def frameStart : Nat := 288960
def rule : BoundRule := .sum [.predecessor 0 289070 .coefficient, .predecessor 1 289071 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 289070 .coefficient)
      LeftBound289068.bound (LeftBound289068.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1129.exact289069RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound289068.bound, RecordedBoundRefines] <;> decide)
      (LeftBound289068.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 289071 .coefficient)
      LeftBound289049.bound (LeftBound289049.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1129.exact289054RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound289049.bound, RecordedBoundRefines] <;> decide)
      (LeftBound289049.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound289068.bound, LeftBound289049.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289068.bound, LeftBound289049.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound289068.actual selector witness, LeftBound289049.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289072

namespace LeftBound289085
def owner : Owner := ⟨.program ⟨257⟩, ⟨17295⟩⟩
def transferEvent : Nat := 289085
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 289083 .coefficient, .predecessor 1 289084 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 289083 .coefficient)
      LeftBound288908.bound (LeftBound288908.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1129.exact289082RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound288908.bound, RecordedBoundRefines] <;> decide)
      (LeftBound288908.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 289084 .coefficient)
      LeftBound288891.bound (LeftBound288891.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1128.exact288898RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound288891.bound, RecordedBoundRefines] <;> decide)
      (LeftBound288891.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound288908.bound, LeftBound288891.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound288908.bound, LeftBound288891.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound288908.actual selector witness, LeftBound288891.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289085

namespace LeftBound289088
def owner : Owner := ⟨.program ⟨257⟩, ⟨17295⟩⟩
def transferEvent : Nat := 289088
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 289082 .summary, .result 288898 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 289082 .summary)
      LeftBound288910.bound (LeftBound288910.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨16232⟩⟩) (rawTerms := some (Proof.Events1129.exact289082RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound288910.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 288898 .summary)
      LeftBound288893.bound (LeftBound288893.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17294⟩⟩) (rawTerms := some (Proof.Events1128.exact288898RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound288893.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound288910.bound, LeftBound288893.bound]
def bound : CoeffClass := .finite ⟨2997816280693142192128, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound288910.bound, LeftBound288893.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound288910.actual selector witness, LeftBound288893.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289088

namespace LeftBound289092
def owner : Owner := ⟨.program ⟨257⟩, ⟨17595⟩⟩
def transferEvent : Nat := 289092
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 289090 .coefficient) (.predecessor 1 289091 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 289090 .coefficient)
      LeftBound289085.bound (LeftBound289085.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1129.exact289089RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound289085.bound, RecordedBoundRefines] <;> decide)
      (LeftBound289085.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 289091 .coefficient)
      LeftAuthority288813.bound (LeftAuthority288813.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1128.exact288814RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority288813.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority288813.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound289085.bound LeftAuthority288813.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289085.bound, LeftAuthority288813.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound289085.actual selector witness) * (LeftAuthority288813.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound289092

namespace LeftBound289093
def owner : Owner := ⟨.program ⟨257⟩, ⟨17595⟩⟩
def transferEvent : Nat := 289093
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨17593⟩⟩]⟩ [⟨.result 288814 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 288814 .coefficient)
      LeftAuthority288813.bound (LeftAuthority288813.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨17593⟩⟩) (rawTerms := some (Proof.Events1128.exact288814RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority288813.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority288813.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority288813.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority288813.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority288813.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound289093

namespace LeftBound289094
def owner : Owner := ⟨.program ⟨257⟩, ⟨17595⟩⟩
def transferEvent : Nat := 289094
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 289089 .summary) (.transfer 289093) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 289089 .summary)
      LeftBound289088.bound (LeftBound289088.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17295⟩⟩) (rawTerms := some (Proof.Events1129.exact289089RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound289088.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 289093)
      LeftBound289093.bound (LeftBound289093.actual selector witness) := by
  exact .transfer (LeftBound289093.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound289088.bound LeftBound289093.bound
def bound : CoeffClass := .finite ⟨32188807212483504816668771614720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289088.bound, LeftBound289093.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound289088.actual selector witness) * (LeftBound289093.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound289094

namespace LeftBound289105
def owner : Owner := ⟨.program ⟨257⟩, ⟨16478⟩⟩
def transferEvent : Nat := 289105
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 289103 .coefficient) (.value (.predecessor 1 289104 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 289103 .coefficient)
      LeftAuthority289101.bound (LeftAuthority289101.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1129.exact289102RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority289101.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority289101.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 289104 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority289101.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority289101.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority289101.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound289105

namespace LeftBound289109
def owner : Owner := ⟨.program ⟨257⟩, ⟨16479⟩⟩
def transferEvent : Nat := 289109
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 289107 .coefficient) (.predecessor 1 289108 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 289107 .coefficient)
      LeftBound280742.bound (LeftBound280742.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1096.exact280745RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280742.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280742.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 289108 .coefficient)
      LeftBound289105.bound (LeftBound289105.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1129.exact289106RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound289105.bound, RecordedBoundRefines] <;> decide)
      (LeftBound289105.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound280742.bound LeftBound289105.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280742.bound, LeftBound289105.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound280742.actual selector witness) * (LeftBound289105.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound289109

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
