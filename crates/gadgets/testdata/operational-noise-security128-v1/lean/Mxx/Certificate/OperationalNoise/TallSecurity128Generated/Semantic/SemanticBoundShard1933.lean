import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1899
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1932

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound285179
def owner : Owner := ⟨.program ⟨257⟩, ⟨64184⟩⟩
def transferEvent : Nat := 285179
def frameStart : Nat := 285120
def rule : BoundRule := .product (.predecessor 0 285177 .coefficient) (.predecessor 1 285178 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 285177 .coefficient)
      LeftAuthority285175.bound (LeftAuthority285175.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1113.exact285176RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority285175.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority285175.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 285178 .coefficient)
      LeftBound285173.bound (LeftBound285173.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1113.exact285174RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound285173.bound, RecordedBoundRefines] <;> decide)
      (LeftBound285173.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority285175.bound LeftBound285173.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority285175.bound, LeftBound285173.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority285175.actual selector witness) * (LeftBound285173.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound285179

namespace LeftBound285193
def owner : Owner := ⟨.program ⟨257⟩, ⟨9539⟩⟩
def transferEvent : Nat := 285193
def frameStart : Nat := 285120
def rule : BoundRule := .scale (.predecessor 0 285191 .coefficient) (.value (.predecessor 1 285192 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 285191 .coefficient)
      LeftAuthority285189.bound (LeftAuthority285189.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1114.exact285190RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority285189.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority285189.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 285192 .coefficient)
      LeftAuthority285123.bound (LeftAuthority285123.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority285123.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority285189.bound LeftAuthority285123.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority285189.bound, LeftAuthority285123.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority285189.actual selector witness) * (LeftAuthority285123.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound285193

namespace LeftBound285196
def owner : Owner := ⟨.program ⟨257⟩, ⟨7293⟩⟩
def transferEvent : Nat := 285196
def frameStart : Nat := 285120
def rule : BoundRule := .identity (.predecessor 0 285195 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 285195 .coefficient)
      LeftAuthority285183.bound (LeftAuthority285183.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1114.exact285184RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority285183.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority285183.derived selector witness)

def rawBound : CoeffClass := LeftAuthority285183.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority285183.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority285183.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound285196

namespace LeftBound285200
def owner : Owner := ⟨.program ⟨257⟩, ⟨9540⟩⟩
def transferEvent : Nat := 285200
def frameStart : Nat := 285120
def rule : BoundRule := .product (.predecessor 0 285198 .coefficient) (.predecessor 1 285199 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 285198 .coefficient)
      LeftBound285196.bound (LeftBound285196.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1114.exact285197RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound285196.bound, RecordedBoundRefines] <;> decide)
      (LeftBound285196.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 285199 .coefficient)
      LeftBound285193.bound (LeftBound285193.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1114.exact285194RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound285193.bound, RecordedBoundRefines] <;> decide)
      (LeftBound285193.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound285196.bound LeftBound285193.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound285196.bound, LeftBound285193.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound285196.actual selector witness) * (LeftBound285193.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound285200

namespace LeftBound285205
def owner : Owner := ⟨.program ⟨257⟩, ⟨64185⟩⟩
def transferEvent : Nat := 285205
def frameStart : Nat := 285120
def rule : BoundRule := .sum [.predecessor 0 285203 .coefficient, .predecessor 1 285204 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 285203 .coefficient)
      LeftBound285200.bound (LeftBound285200.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1114.exact285202RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound285200.bound, RecordedBoundRefines] <;> decide)
      (LeftBound285200.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 285204 .coefficient)
      LeftBound285179.bound (LeftBound285179.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1113.exact285181RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound285179.bound, RecordedBoundRefines] <;> decide)
      (LeftBound285179.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound285200.bound, LeftBound285179.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound285200.bound, LeftBound285179.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound285200.actual selector witness, LeftBound285179.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound285205

namespace LeftBound285209
def owner : Owner := ⟨.program ⟨257⟩, ⟨64376⟩⟩
def transferEvent : Nat := 285209
def frameStart : Nat := 285120
def rule : BoundRule := .product (.predecessor 0 285207 .coefficient) (.predecessor 1 285208 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 285207 .coefficient)
      LeftBound285205.bound (LeftBound285205.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1114.exact285206RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound285205.bound, RecordedBoundRefines] <;> decide)
      (LeftBound285205.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 285208 .coefficient)
      LeftAuthority285164.bound (LeftAuthority285164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1113.exact285165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority285164.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority285164.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound285205.bound LeftAuthority285164.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound285205.bound, LeftAuthority285164.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound285205.actual selector witness) * (LeftAuthority285164.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound285209

namespace LeftBound285220
def owner : Owner := ⟨.program ⟨257⟩, ⟨62762⟩⟩
def transferEvent : Nat := 285220
def frameStart : Nat := 285120
def rule : BoundRule := .product (.predecessor 0 285218 .coefficient) (.predecessor 1 285219 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 285218 .coefficient)
      LeftAuthority285175.bound (LeftAuthority285175.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1113.exact285176RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority285175.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority285175.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 285219 .coefficient)
      LeftAuthority285216.bound (LeftAuthority285216.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1114.exact285217RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority285216.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority285216.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority285175.bound LeftAuthority285216.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority285175.bound, LeftAuthority285216.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority285175.actual selector witness) * (LeftAuthority285216.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound285220

namespace LeftBound285228
def owner : Owner := ⟨.program ⟨257⟩, ⟨62763⟩⟩
def transferEvent : Nat := 285228
def frameStart : Nat := 285120
def rule : BoundRule := .sum [.predecessor 0 285226 .coefficient, .predecessor 1 285227 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 285226 .coefficient)
      LeftAuthority285224.bound (LeftAuthority285224.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1114.exact285225RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority285224.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority285224.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 285227 .coefficient)
      LeftBound285220.bound (LeftBound285220.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1114.exact285222RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound285220.bound, RecordedBoundRefines] <;> decide)
      (LeftBound285220.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority285224.bound, LeftBound285220.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority285224.bound, LeftBound285220.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority285224.actual selector witness, LeftBound285220.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound285228

namespace LeftBound285232
def owner : Owner := ⟨.program ⟨257⟩, ⟨64377⟩⟩
def transferEvent : Nat := 285232
def frameStart : Nat := 285120
def rule : BoundRule := .sum [.predecessor 0 285230 .coefficient, .predecessor 1 285231 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 285230 .coefficient)
      LeftBound285228.bound (LeftBound285228.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1114.exact285229RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound285228.bound, RecordedBoundRefines] <;> decide)
      (LeftBound285228.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 285231 .coefficient)
      LeftBound285209.bound (LeftBound285209.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1114.exact285214RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound285209.bound, RecordedBoundRefines] <;> decide)
      (LeftBound285209.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound285228.bound, LeftBound285209.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound285228.bound, LeftBound285209.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound285228.actual selector witness, LeftBound285209.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound285232

namespace LeftBound285245
def owner : Owner := ⟨.program ⟨257⟩, ⟨64375⟩⟩
def transferEvent : Nat := 285245
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 285243 .coefficient, .predecessor 1 285244 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 285243 .coefficient)
      LeftBound285068.bound (LeftBound285068.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1114.exact285242RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound285068.bound, RecordedBoundRefines] <;> decide)
      (LeftBound285068.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 285244 .coefficient)
      LeftBound285051.bound (LeftBound285051.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1113.exact285058RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound285051.bound, RecordedBoundRefines] <;> decide)
      (LeftBound285051.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound285068.bound, LeftBound285051.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound285068.bound, LeftBound285051.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound285068.actual selector witness, LeftBound285051.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound285245

namespace LeftBound285248
def owner : Owner := ⟨.program ⟨257⟩, ⟨64375⟩⟩
def transferEvent : Nat := 285248
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 285242 .summary, .result 285058 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 285242 .summary)
      LeftBound285070.bound (LeftBound285070.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨63312⟩⟩) (rawTerms := some (Proof.Events1114.exact285242RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound285070.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 285058 .summary)
      LeftBound285053.bound (LeftBound285053.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64374⟩⟩) (rawTerms := some (Proof.Events1113.exact285058RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound285053.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound285070.bound, LeftBound285053.bound]
def bound : CoeffClass := .finite ⟨2997999239428004118528, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound285070.bound, LeftBound285053.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound285070.actual selector witness, LeftBound285053.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound285248

namespace LeftBound285252
def owner : Owner := ⟨.program ⟨257⟩, ⟨64688⟩⟩
def transferEvent : Nat := 285252
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 285250 .coefficient) (.predecessor 1 285251 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 285250 .coefficient)
      LeftBound285245.bound (LeftBound285245.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1114.exact285249RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound285245.bound, RecordedBoundRefines] <;> decide)
      (LeftBound285245.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 285251 .coefficient)
      LeftAuthority284973.bound (LeftAuthority284973.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1113.exact284974RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority284973.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority284973.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound285245.bound LeftAuthority284973.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound285245.bound, LeftAuthority284973.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound285245.actual selector witness) * (LeftAuthority284973.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound285252

namespace LeftBound285253
def owner : Owner := ⟨.program ⟨257⟩, ⟨64688⟩⟩
def transferEvent : Nat := 285253
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨64686⟩⟩]⟩ [⟨.result 284974 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 284974 .coefficient)
      LeftAuthority284973.bound (LeftAuthority284973.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨64686⟩⟩) (rawTerms := some (Proof.Events1113.exact284974RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority284973.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority284973.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority284973.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority284973.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority284973.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound285253

namespace LeftBound285254
def owner : Owner := ⟨.program ⟨257⟩, ⟨64688⟩⟩
def transferEvent : Nat := 285254
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 285249 .summary) (.transfer 285253) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 285249 .summary)
      LeftBound285248.bound (LeftBound285248.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64375⟩⟩) (rawTerms := some (Proof.Events1114.exact285249RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound285248.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 285253)
      LeftBound285253.bound (LeftBound285253.actual selector witness) := by
  exact .transfer (LeftBound285253.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound285248.bound LeftBound285253.bound
def bound : CoeffClass := .finite ⟨32190771716940378589077669150720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound285248.bound, LeftBound285253.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound285248.actual selector witness) * (LeftBound285253.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound285254

namespace LeftBound285265
def owner : Owner := ⟨.program ⟨257⟩, ⟨63558⟩⟩
def transferEvent : Nat := 285265
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 285263 .coefficient) (.value (.predecessor 1 285264 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 285263 .coefficient)
      LeftAuthority285261.bound (LeftAuthority285261.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1114.exact285262RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority285261.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority285261.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 285264 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority285261.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority285261.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority285261.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound285265

namespace LeftBound285269
def owner : Owner := ⟨.program ⟨257⟩, ⟨63559⟩⟩
def transferEvent : Nat := 285269
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 285267 .coefficient) (.predecessor 1 285268 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 285267 .coefficient)
      LeftBound280742.bound (LeftBound280742.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1096.exact280745RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280742.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280742.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 285268 .coefficient)
      LeftBound285265.bound (LeftBound285265.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1114.exact285266RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound285265.bound, RecordedBoundRefines] <;> decide)
      (LeftBound285265.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound280742.bound LeftBound285265.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280742.bound, LeftBound285265.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound280742.actual selector witness) * (LeftBound285265.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound285269

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
