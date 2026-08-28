import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard903

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound137010
def owner : Owner := ⟨.program ⟨257⟩, ⟨35998⟩⟩
def transferEvent : Nat := 137010
def frameStart : Nat := 136960
def rule : BoundRule := .sum [.predecessor 0 137008 .coefficient, .predecessor 1 137009 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 137008 .coefficient)
      LeftBound136993.bound (LeftBound136993.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound136993.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 137009 .coefficient)
      LeftAuthority137006.bound (LeftAuthority137006.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority137006.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound136993.bound, LeftAuthority137006.bound]
def bound : CoeffClass := .finite ⟨1600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound136993.bound, LeftAuthority137006.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound136993.actual selector witness, LeftAuthority137006.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound137010

namespace LeftBound137013
def owner : Owner := ⟨.program ⟨257⟩, ⟨35999⟩⟩
def transferEvent : Nat := 137013
def frameStart : Nat := 136960
def rule : BoundRule := .identity (.predecessor 0 137012 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 137012 .coefficient)
      LeftBound137010.bound (LeftBound137010.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound137010.derived selector witness)

def rawBound : CoeffClass := LeftBound137010.bound
def bound : CoeffClass := .finite ⟨1600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound137010.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound137010.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound137013

namespace LeftBound137019
def owner : Owner := ⟨.program ⟨257⟩, ⟨36000⟩⟩
def transferEvent : Nat := 137019
def frameStart : Nat := 136960
def rule : BoundRule := .product (.predecessor 0 137017 .coefficient) (.predecessor 1 137018 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 137017 .coefficient)
      LeftAuthority137015.bound (LeftAuthority137015.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events535.exact137016RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority137015.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority137015.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 137018 .coefficient)
      LeftBound137013.bound (LeftBound137013.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events535.exact137014RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound137013.bound, RecordedBoundRefines] <;> decide)
      (LeftBound137013.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority137015.bound LeftBound137013.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority137015.bound, LeftBound137013.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority137015.actual selector witness) * (LeftBound137013.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound137019

namespace LeftBound137035
def owner : Owner := ⟨.program ⟨257⟩, ⟨9551⟩⟩
def transferEvent : Nat := 137035
def frameStart : Nat := 136960
def rule : BoundRule := .scale (.predecessor 0 137033 .coefficient) (.value (.predecessor 1 137034 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 137033 .coefficient)
      LeftAuthority137031.bound (LeftAuthority137031.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events535.exact137032RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority137031.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority137031.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 137034 .coefficient)
      LeftAuthority137022.bound (LeftAuthority137022.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority137022.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority137031.bound LeftAuthority137022.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority137031.bound, LeftAuthority137022.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority137031.actual selector witness) * (LeftAuthority137022.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound137035

namespace LeftBound137038
def owner : Owner := ⟨.program ⟨257⟩, ⟨7297⟩⟩
def transferEvent : Nat := 137038
def frameStart : Nat := 136960
def rule : BoundRule := .identity (.predecessor 0 137037 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 137037 .coefficient)
      LeftAuthority137025.bound (LeftAuthority137025.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events535.exact137026RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority137025.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority137025.derived selector witness)

def rawBound : CoeffClass := LeftAuthority137025.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority137025.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority137025.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound137038

namespace LeftBound137042
def owner : Owner := ⟨.program ⟨257⟩, ⟨9552⟩⟩
def transferEvent : Nat := 137042
def frameStart : Nat := 136960
def rule : BoundRule := .product (.predecessor 0 137040 .coefficient) (.predecessor 1 137041 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 137040 .coefficient)
      LeftBound137038.bound (LeftBound137038.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events535.exact137039RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound137038.bound, RecordedBoundRefines] <;> decide)
      (LeftBound137038.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 137041 .coefficient)
      LeftBound137035.bound (LeftBound137035.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events535.exact137036RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound137035.bound, RecordedBoundRefines] <;> decide)
      (LeftBound137035.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound137038.bound LeftBound137035.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound137038.bound, LeftBound137035.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound137038.actual selector witness) * (LeftBound137035.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound137042

namespace LeftBound137047
def owner : Owner := ⟨.program ⟨257⟩, ⟨36001⟩⟩
def transferEvent : Nat := 137047
def frameStart : Nat := 136960
def rule : BoundRule := .sum [.predecessor 0 137045 .coefficient, .predecessor 1 137046 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 137045 .coefficient)
      LeftBound137042.bound (LeftBound137042.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events535.exact137044RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound137042.bound, RecordedBoundRefines] <;> decide)
      (LeftBound137042.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 137046 .coefficient)
      LeftBound137019.bound (LeftBound137019.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events535.exact137021RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound137019.bound, RecordedBoundRefines] <;> decide)
      (LeftBound137019.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound137042.bound, LeftBound137019.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound137042.bound, LeftBound137019.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound137042.actual selector witness, LeftBound137019.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound137047

namespace LeftBound137051
def owner : Owner := ⟨.program ⟨257⟩, ⟨36185⟩⟩
def transferEvent : Nat := 137051
def frameStart : Nat := 136960
def rule : BoundRule := .product (.predecessor 0 137049 .coefficient) (.predecessor 1 137050 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 137049 .coefficient)
      LeftBound137047.bound (LeftBound137047.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events535.exact137048RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound137047.bound, RecordedBoundRefines] <;> decide)
      (LeftBound137047.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 137050 .coefficient)
      LeftAuthority137004.bound (LeftAuthority137004.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events535.exact137005RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority137004.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority137004.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound137047.bound LeftAuthority137004.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound137047.bound, LeftAuthority137004.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound137047.actual selector witness) * (LeftAuthority137004.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound137051

namespace LeftBound137062
def owner : Owner := ⟨.program ⟨257⟩, ⟨34694⟩⟩
def transferEvent : Nat := 137062
def frameStart : Nat := 136960
def rule : BoundRule := .product (.predecessor 0 137060 .coefficient) (.predecessor 1 137061 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 137060 .coefficient)
      LeftAuthority137015.bound (LeftAuthority137015.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events535.exact137016RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority137015.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority137015.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 137061 .coefficient)
      LeftAuthority137058.bound (LeftAuthority137058.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events535.exact137059RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority137058.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority137058.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority137015.bound LeftAuthority137058.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority137015.bound, LeftAuthority137058.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority137015.actual selector witness) * (LeftAuthority137058.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound137062

namespace LeftBound137070
def owner : Owner := ⟨.program ⟨257⟩, ⟨34695⟩⟩
def transferEvent : Nat := 137070
def frameStart : Nat := 136960
def rule : BoundRule := .sum [.predecessor 0 137068 .coefficient, .predecessor 1 137069 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 137068 .coefficient)
      LeftAuthority137066.bound (LeftAuthority137066.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events535.exact137067RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority137066.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority137066.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 137069 .coefficient)
      LeftBound137062.bound (LeftBound137062.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events535.exact137064RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound137062.bound, RecordedBoundRefines] <;> decide)
      (LeftBound137062.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority137066.bound, LeftBound137062.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority137066.bound, LeftBound137062.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority137066.actual selector witness, LeftBound137062.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound137070

namespace LeftBound137074
def owner : Owner := ⟨.program ⟨257⟩, ⟨36186⟩⟩
def transferEvent : Nat := 137074
def frameStart : Nat := 136960
def rule : BoundRule := .sum [.predecessor 0 137072 .coefficient, .predecessor 1 137073 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 137072 .coefficient)
      LeftBound137070.bound (LeftBound137070.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events535.exact137071RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound137070.bound, RecordedBoundRefines] <;> decide)
      (LeftBound137070.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 137073 .coefficient)
      LeftBound137051.bound (LeftBound137051.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events535.exact137056RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound137051.bound, RecordedBoundRefines] <;> decide)
      (LeftBound137051.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound137070.bound, LeftBound137051.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound137070.bound, LeftBound137051.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound137070.actual selector witness, LeftBound137051.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound137074

namespace LeftBound137087
def owner : Owner := ⟨.program ⟨257⟩, ⟨36184⟩⟩
def transferEvent : Nat := 137087
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 137085 .coefficient, .predecessor 1 137086 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 137085 .coefficient)
      LeftBound136908.bound (LeftBound136908.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events535.exact137084RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound136908.bound, RecordedBoundRefines] <;> decide)
      (LeftBound136908.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 137086 .coefficient)
      LeftBound136891.bound (LeftBound136891.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events534.exact136898RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound136891.bound, RecordedBoundRefines] <;> decide)
      (LeftBound136891.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound136908.bound, LeftBound136891.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound136908.bound, LeftBound136891.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound136908.actual selector witness, LeftBound136891.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound137087

namespace LeftBound137090
def owner : Owner := ⟨.program ⟨257⟩, ⟨36184⟩⟩
def transferEvent : Nat := 137090
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 137084 .summary, .result 136898 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 137084 .summary)
      LeftBound136910.bound (LeftBound136910.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨35122⟩⟩) (rawTerms := some (Proof.Events535.exact137084RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound136910.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 136898 .summary)
      LeftBound136893.bound (LeftBound136893.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36183⟩⟩) (rawTerms := some (Proof.Events534.exact136898RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound136893.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound136910.bound, LeftBound136893.bound]
def bound : CoeffClass := .finite ⟨2998163902289379852288, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound136910.bound, LeftBound136893.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound136910.actual selector witness, LeftBound136893.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound137090

namespace LeftBound137094
def owner : Owner := ⟨.program ⟨257⟩, ⟨36456⟩⟩
def transferEvent : Nat := 137094
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 137092 .coefficient) (.predecessor 1 137093 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 137092 .coefficient)
      LeftBound137087.bound (LeftBound137087.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events535.exact137091RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound137087.bound, RecordedBoundRefines] <;> decide)
      (LeftBound137087.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 137093 .coefficient)
      LeftAuthority136813.bound (LeftAuthority136813.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events534.exact136814RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority136813.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority136813.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound137087.bound LeftAuthority136813.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound137087.bound, LeftAuthority136813.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound137087.actual selector witness) * (LeftAuthority136813.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound137094

namespace LeftBound137095
def owner : Owner := ⟨.program ⟨257⟩, ⟨36456⟩⟩
def transferEvent : Nat := 137095
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨36454⟩⟩]⟩ [⟨.result 136814 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 136814 .coefficient)
      LeftAuthority136813.bound (LeftAuthority136813.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨36454⟩⟩) (rawTerms := some (Proof.Events534.exact136814RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority136813.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority136813.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority136813.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority136813.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority136813.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound137095

namespace LeftBound137096
def owner : Owner := ⟨.program ⟨257⟩, ⟨36456⟩⟩
def transferEvent : Nat := 137096
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 137091 .summary) (.transfer 137095) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 137091 .summary)
      LeftBound137090.bound (LeftBound137090.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36184⟩⟩) (rawTerms := some (Proof.Events535.exact137091RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound137090.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 137095)
      LeftBound137095.bound (LeftBound137095.actual selector witness) := by
  exact .transfer (LeftBound137095.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound137090.bound LeftBound137095.bound
def bound : CoeffClass := .finite ⟨32192539770951564984245676933120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound137090.bound, LeftBound137095.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound137090.actual selector witness) * (LeftBound137095.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound137096

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
