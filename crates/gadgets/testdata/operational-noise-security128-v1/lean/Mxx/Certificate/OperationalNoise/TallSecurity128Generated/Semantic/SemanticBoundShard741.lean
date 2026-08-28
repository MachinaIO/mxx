import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard740

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound113062
def owner : Owner := ⟨.program ⟨257⟩, ⟨19990⟩⟩
def transferEvent : Nat := 113062
def frameStart : Nat := 113012
def rule : BoundRule := .sum [.predecessor 0 113060 .coefficient, .predecessor 1 113061 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 113060 .coefficient)
      LeftBound113045.bound (LeftBound113045.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound113045.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 113061 .coefficient)
      LeftAuthority113058.bound (LeftAuthority113058.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority113058.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound113045.bound, LeftAuthority113058.bound]
def bound : CoeffClass := .finite ⟨9, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound113045.bound, LeftAuthority113058.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound113045.actual selector witness, LeftAuthority113058.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound113062

namespace LeftBound113065
def owner : Owner := ⟨.program ⟨257⟩, ⟨19991⟩⟩
def transferEvent : Nat := 113065
def frameStart : Nat := 113012
def rule : BoundRule := .identity (.predecessor 0 113064 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 113064 .coefficient)
      LeftBound113062.bound (LeftBound113062.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound113062.derived selector witness)

def rawBound : CoeffClass := LeftBound113062.bound
def bound : CoeffClass := .finite ⟨9, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound113062.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound113062.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound113065

namespace LeftBound113071
def owner : Owner := ⟨.program ⟨257⟩, ⟨19992⟩⟩
def transferEvent : Nat := 113071
def frameStart : Nat := 113012
def rule : BoundRule := .product (.predecessor 0 113069 .coefficient) (.predecessor 1 113070 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 113069 .coefficient)
      LeftAuthority113067.bound (LeftAuthority113067.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events441.exact113068RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority113067.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority113067.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 113070 .coefficient)
      LeftBound113065.bound (LeftBound113065.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events441.exact113066RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound113065.bound, RecordedBoundRefines] <;> decide)
      (LeftBound113065.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority113067.bound LeftBound113065.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority113067.bound, LeftBound113065.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority113067.actual selector witness) * (LeftBound113065.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound113071

namespace LeftBound113087
def owner : Owner := ⟨.program ⟨257⟩, ⟨9572⟩⟩
def transferEvent : Nat := 113087
def frameStart : Nat := 113012
def rule : BoundRule := .scale (.predecessor 0 113085 .coefficient) (.value (.predecessor 1 113086 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 113085 .coefficient)
      LeftAuthority113083.bound (LeftAuthority113083.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events441.exact113084RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority113083.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority113083.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 113086 .coefficient)
      LeftAuthority113074.bound (LeftAuthority113074.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority113074.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority113083.bound LeftAuthority113074.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority113083.bound, LeftAuthority113074.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority113083.actual selector witness) * (LeftAuthority113074.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound113087

namespace LeftBound113090
def owner : Owner := ⟨.program ⟨257⟩, ⟨7277⟩⟩
def transferEvent : Nat := 113090
def frameStart : Nat := 113012
def rule : BoundRule := .identity (.predecessor 0 113089 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 113089 .coefficient)
      LeftAuthority113077.bound (LeftAuthority113077.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events441.exact113078RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority113077.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority113077.derived selector witness)

def rawBound : CoeffClass := LeftAuthority113077.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority113077.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority113077.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound113090

namespace LeftBound113094
def owner : Owner := ⟨.program ⟨257⟩, ⟨9573⟩⟩
def transferEvent : Nat := 113094
def frameStart : Nat := 113012
def rule : BoundRule := .product (.predecessor 0 113092 .coefficient) (.predecessor 1 113093 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 113092 .coefficient)
      LeftBound113090.bound (LeftBound113090.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events441.exact113091RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound113090.bound, RecordedBoundRefines] <;> decide)
      (LeftBound113090.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 113093 .coefficient)
      LeftBound113087.bound (LeftBound113087.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events441.exact113088RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound113087.bound, RecordedBoundRefines] <;> decide)
      (LeftBound113087.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound113090.bound LeftBound113087.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound113090.bound, LeftBound113087.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound113090.actual selector witness) * (LeftBound113087.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound113094

namespace LeftBound113099
def owner : Owner := ⟨.program ⟨257⟩, ⟨19993⟩⟩
def transferEvent : Nat := 113099
def frameStart : Nat := 113012
def rule : BoundRule := .sum [.predecessor 0 113097 .coefficient, .predecessor 1 113098 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 113097 .coefficient)
      LeftBound113094.bound (LeftBound113094.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events441.exact113096RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound113094.bound, RecordedBoundRefines] <;> decide)
      (LeftBound113094.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 113098 .coefficient)
      LeftBound113071.bound (LeftBound113071.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events441.exact113073RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound113071.bound, RecordedBoundRefines] <;> decide)
      (LeftBound113071.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound113094.bound, LeftBound113071.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound113094.bound, LeftBound113071.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound113094.actual selector witness, LeftBound113071.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound113099

namespace LeftBound113103
def owner : Owner := ⟨.program ⟨257⟩, ⟨20233⟩⟩
def transferEvent : Nat := 113103
def frameStart : Nat := 113012
def rule : BoundRule := .product (.predecessor 0 113101 .coefficient) (.predecessor 1 113102 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 113101 .coefficient)
      LeftBound113099.bound (LeftBound113099.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events441.exact113100RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound113099.bound, RecordedBoundRefines] <;> decide)
      (LeftBound113099.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 113102 .coefficient)
      LeftAuthority113056.bound (LeftAuthority113056.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events441.exact113057RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority113056.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority113056.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound113099.bound LeftAuthority113056.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound113099.bound, LeftAuthority113056.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound113099.actual selector witness) * (LeftAuthority113056.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound113103

namespace LeftBound113114
def owner : Owner := ⟨.program ⟨257⟩, ⟨18598⟩⟩
def transferEvent : Nat := 113114
def frameStart : Nat := 113012
def rule : BoundRule := .product (.predecessor 0 113112 .coefficient) (.predecessor 1 113113 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 113112 .coefficient)
      LeftAuthority113067.bound (LeftAuthority113067.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events441.exact113068RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority113067.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority113067.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 113113 .coefficient)
      LeftAuthority113110.bound (LeftAuthority113110.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events441.exact113111RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority113110.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority113110.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority113067.bound LeftAuthority113110.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority113067.bound, LeftAuthority113110.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority113067.actual selector witness) * (LeftAuthority113110.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound113114

namespace LeftBound113122
def owner : Owner := ⟨.program ⟨257⟩, ⟨18599⟩⟩
def transferEvent : Nat := 113122
def frameStart : Nat := 113012
def rule : BoundRule := .sum [.predecessor 0 113120 .coefficient, .predecessor 1 113121 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 113120 .coefficient)
      LeftAuthority113118.bound (LeftAuthority113118.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events441.exact113119RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority113118.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority113118.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 113121 .coefficient)
      LeftBound113114.bound (LeftBound113114.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events441.exact113116RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound113114.bound, RecordedBoundRefines] <;> decide)
      (LeftBound113114.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority113118.bound, LeftBound113114.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority113118.bound, LeftBound113114.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority113118.actual selector witness, LeftBound113114.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound113122

namespace LeftBound113126
def owner : Owner := ⟨.program ⟨257⟩, ⟨20234⟩⟩
def transferEvent : Nat := 113126
def frameStart : Nat := 113012
def rule : BoundRule := .sum [.predecessor 0 113124 .coefficient, .predecessor 1 113125 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 113124 .coefficient)
      LeftBound113122.bound (LeftBound113122.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events441.exact113123RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound113122.bound, RecordedBoundRefines] <;> decide)
      (LeftBound113122.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 113125 .coefficient)
      LeftBound113103.bound (LeftBound113103.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events441.exact113108RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound113103.bound, RecordedBoundRefines] <;> decide)
      (LeftBound113103.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound113122.bound, LeftBound113103.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound113122.bound, LeftBound113103.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound113122.actual selector witness, LeftBound113103.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound113126

namespace LeftBound113139
def owner : Owner := ⟨.program ⟨257⟩, ⟨20232⟩⟩
def transferEvent : Nat := 113139
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 113137 .coefficient, .predecessor 1 113138 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 113137 .coefficient)
      LeftBound112960.bound (LeftBound112960.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events441.exact113136RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound112960.bound, RecordedBoundRefines] <;> decide)
      (LeftBound112960.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 113138 .coefficient)
      LeftBound112943.bound (LeftBound112943.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events441.exact112950RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound112943.bound, RecordedBoundRefines] <;> decide)
      (LeftBound112943.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound112960.bound, LeftBound112943.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound112960.bound, LeftBound112943.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound112960.actual selector witness, LeftBound112943.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound113139

namespace LeftBound113142
def owner : Owner := ⟨.program ⟨257⟩, ⟨20232⟩⟩
def transferEvent : Nat := 113142
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 113136 .summary, .result 112950 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 113136 .summary)
      LeftBound112962.bound (LeftBound112962.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨19162⟩⟩) (rawTerms := some (Proof.Events441.exact113136RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound112962.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 112950 .summary)
      LeftBound112945.bound (LeftBound112945.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20231⟩⟩) (rawTerms := some (Proof.Events441.exact112950RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound112945.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound112962.bound, LeftBound112945.bound]
def bound : CoeffClass := .finite ⟨2997825428629885288448, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound112962.bound, LeftBound112945.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound112962.actual selector witness, LeftBound112945.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound113142

namespace LeftBound113146
def owner : Owner := ⟨.program ⟨257⟩, ⟨20685⟩⟩
def transferEvent : Nat := 113146
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 113144 .coefficient) (.predecessor 1 113145 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 113144 .coefficient)
      LeftBound113139.bound (LeftBound113139.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events441.exact113143RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound113139.bound, RecordedBoundRefines] <;> decide)
      (LeftBound113139.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 113145 .coefficient)
      LeftAuthority112865.bound (LeftAuthority112865.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events440.exact112866RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority112865.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority112865.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound113139.bound LeftAuthority112865.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound113139.bound, LeftAuthority112865.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound113139.actual selector witness) * (LeftAuthority112865.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound113146

namespace LeftBound113147
def owner : Owner := ⟨.program ⟨257⟩, ⟨20685⟩⟩
def transferEvent : Nat := 113147
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨20683⟩⟩]⟩ [⟨.result 112866 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 112866 .coefficient)
      LeftAuthority112865.bound (LeftAuthority112865.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨20683⟩⟩) (rawTerms := some (Proof.Events440.exact112866RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority112865.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority112865.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority112865.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority112865.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority112865.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound113147

namespace LeftBound113148
def owner : Owner := ⟨.program ⟨257⟩, ⟨20685⟩⟩
def transferEvent : Nat := 113148
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 113143 .summary) (.transfer 113147) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 113143 .summary)
      LeftBound113142.bound (LeftBound113142.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20232⟩⟩) (rawTerms := some (Proof.Events441.exact113143RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound113142.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 113147)
      LeftBound113147.bound (LeftBound113147.actual selector witness) := by
  exact .transfer (LeftBound113147.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound113142.bound LeftBound113147.bound
def bound : CoeffClass := .finite ⟨32188905437706348505289216491520, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound113142.bound, LeftBound113147.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound113142.actual selector witness) * (LeftBound113147.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound113148

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
