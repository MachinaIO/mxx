import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1113

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound167203
def owner : Owner := ⟨.program ⟨257⟩, ⟨26191⟩⟩
def transferEvent : Nat := 167203
def frameStart : Nat := 167174
def rule : BoundRule := .product (.predecessor 0 167201 .coefficient) (.predecessor 1 167202 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 167201 .coefficient)
      LeftAuthority167199.bound (LeftAuthority167199.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events653.exact167200RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority167199.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority167199.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 167202 .coefficient)
      LeftAuthority167196.bound (LeftAuthority167196.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events653.exact167197RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority167196.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority167196.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority167199.bound LeftAuthority167196.bound
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority167199.bound, LeftAuthority167196.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority167199.actual selector witness) * (LeftAuthority167196.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound167203

namespace LeftBound167207
def owner : Owner := ⟨.program ⟨257⟩, ⟨26192⟩⟩
def transferEvent : Nat := 167207
def frameStart : Nat := 167174
def rule : BoundRule := .identity (.predecessor 0 167206 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 167206 .coefficient)
      LeftBound167203.bound (LeftBound167203.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events653.exact167205RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound167203.bound, RecordedBoundRefines] <;> decide)
      (LeftBound167203.derived selector witness)

def rawBound : CoeffClass := LeftBound167203.bound
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound167203.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound167203.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound167207

namespace LeftBound167224
def owner : Owner := ⟨.program ⟨257⟩, ⟨27702⟩⟩
def transferEvent : Nat := 167224
def frameStart : Nat := 167174
def rule : BoundRule := .sum [.predecessor 0 167222 .coefficient, .predecessor 1 167223 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 167222 .coefficient)
      LeftBound167207.bound (LeftBound167207.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound167207.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 167223 .coefficient)
      LeftAuthority167220.bound (LeftAuthority167220.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority167220.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound167207.bound, LeftAuthority167220.bound]
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound167207.bound, LeftAuthority167220.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound167207.actual selector witness, LeftAuthority167220.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound167224

namespace LeftBound167227
def owner : Owner := ⟨.program ⟨257⟩, ⟨27703⟩⟩
def transferEvent : Nat := 167227
def frameStart : Nat := 167174
def rule : BoundRule := .identity (.predecessor 0 167226 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 167226 .coefficient)
      LeftBound167224.bound (LeftBound167224.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound167224.derived selector witness)

def rawBound : CoeffClass := LeftBound167224.bound
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound167224.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound167224.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound167227

namespace LeftBound167233
def owner : Owner := ⟨.program ⟨257⟩, ⟨27704⟩⟩
def transferEvent : Nat := 167233
def frameStart : Nat := 167174
def rule : BoundRule := .product (.predecessor 0 167231 .coefficient) (.predecessor 1 167232 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 167231 .coefficient)
      LeftAuthority167229.bound (LeftAuthority167229.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events653.exact167230RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority167229.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority167229.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 167232 .coefficient)
      LeftBound167227.bound (LeftBound167227.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events653.exact167228RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound167227.bound, RecordedBoundRefines] <;> decide)
      (LeftBound167227.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority167229.bound LeftBound167227.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority167229.bound, LeftBound167227.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority167229.actual selector witness) * (LeftBound167227.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound167233

namespace LeftBound167249
def owner : Owner := ⟨.program ⟨257⟩, ⟨9545⟩⟩
def transferEvent : Nat := 167249
def frameStart : Nat := 167174
def rule : BoundRule := .scale (.predecessor 0 167247 .coefficient) (.value (.predecessor 1 167248 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 167247 .coefficient)
      LeftAuthority167245.bound (LeftAuthority167245.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events653.exact167246RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority167245.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority167245.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 167248 .coefficient)
      LeftAuthority167236.bound (LeftAuthority167236.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority167236.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority167245.bound LeftAuthority167236.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority167245.bound, LeftAuthority167236.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority167245.actual selector witness) * (LeftAuthority167236.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound167249

namespace LeftBound167252
def owner : Owner := ⟨.program ⟨257⟩, ⟨7295⟩⟩
def transferEvent : Nat := 167252
def frameStart : Nat := 167174
def rule : BoundRule := .identity (.predecessor 0 167251 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 167251 .coefficient)
      LeftAuthority167239.bound (LeftAuthority167239.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events653.exact167240RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority167239.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority167239.derived selector witness)

def rawBound : CoeffClass := LeftAuthority167239.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority167239.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority167239.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound167252

namespace LeftBound167256
def owner : Owner := ⟨.program ⟨257⟩, ⟨9546⟩⟩
def transferEvent : Nat := 167256
def frameStart : Nat := 167174
def rule : BoundRule := .product (.predecessor 0 167254 .coefficient) (.predecessor 1 167255 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 167254 .coefficient)
      LeftBound167252.bound (LeftBound167252.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events653.exact167253RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound167252.bound, RecordedBoundRefines] <;> decide)
      (LeftBound167252.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 167255 .coefficient)
      LeftBound167249.bound (LeftBound167249.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events653.exact167250RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound167249.bound, RecordedBoundRefines] <;> decide)
      (LeftBound167249.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound167252.bound LeftBound167249.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound167252.bound, LeftBound167249.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound167252.actual selector witness) * (LeftBound167249.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound167256

namespace LeftBound167261
def owner : Owner := ⟨.program ⟨257⟩, ⟨27705⟩⟩
def transferEvent : Nat := 167261
def frameStart : Nat := 167174
def rule : BoundRule := .sum [.predecessor 0 167259 .coefficient, .predecessor 1 167260 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 167259 .coefficient)
      LeftBound167256.bound (LeftBound167256.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events653.exact167258RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound167256.bound, RecordedBoundRefines] <;> decide)
      (LeftBound167256.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 167260 .coefficient)
      LeftBound167233.bound (LeftBound167233.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events653.exact167235RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound167233.bound, RecordedBoundRefines] <;> decide)
      (LeftBound167233.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound167256.bound, LeftBound167233.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound167256.bound, LeftBound167233.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound167256.actual selector witness, LeftBound167233.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound167261

namespace LeftBound167265
def owner : Owner := ⟨.program ⟨257⟩, ⟨27966⟩⟩
def transferEvent : Nat := 167265
def frameStart : Nat := 167174
def rule : BoundRule := .product (.predecessor 0 167263 .coefficient) (.predecessor 1 167264 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 167263 .coefficient)
      LeftBound167261.bound (LeftBound167261.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events653.exact167262RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound167261.bound, RecordedBoundRefines] <;> decide)
      (LeftBound167261.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 167264 .coefficient)
      LeftAuthority167218.bound (LeftAuthority167218.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events653.exact167219RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority167218.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority167218.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound167261.bound LeftAuthority167218.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound167261.bound, LeftAuthority167218.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound167261.actual selector witness) * (LeftAuthority167218.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound167265

namespace LeftBound167276
def owner : Owner := ⟨.program ⟨257⟩, ⟨26442⟩⟩
def transferEvent : Nat := 167276
def frameStart : Nat := 167174
def rule : BoundRule := .product (.predecessor 0 167274 .coefficient) (.predecessor 1 167275 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 167274 .coefficient)
      LeftAuthority167229.bound (LeftAuthority167229.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events653.exact167230RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority167229.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority167229.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 167275 .coefficient)
      LeftAuthority167272.bound (LeftAuthority167272.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events653.exact167273RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority167272.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority167272.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority167229.bound LeftAuthority167272.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority167229.bound, LeftAuthority167272.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority167229.actual selector witness) * (LeftAuthority167272.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound167276

namespace LeftBound167284
def owner : Owner := ⟨.program ⟨257⟩, ⟨26443⟩⟩
def transferEvent : Nat := 167284
def frameStart : Nat := 167174
def rule : BoundRule := .sum [.predecessor 0 167282 .coefficient, .predecessor 1 167283 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 167282 .coefficient)
      LeftAuthority167280.bound (LeftAuthority167280.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events653.exact167281RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority167280.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority167280.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 167283 .coefficient)
      LeftBound167276.bound (LeftBound167276.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events653.exact167278RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound167276.bound, RecordedBoundRefines] <;> decide)
      (LeftBound167276.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority167280.bound, LeftBound167276.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority167280.bound, LeftBound167276.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority167280.actual selector witness, LeftBound167276.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound167284

namespace LeftBound167288
def owner : Owner := ⟨.program ⟨257⟩, ⟨27967⟩⟩
def transferEvent : Nat := 167288
def frameStart : Nat := 167174
def rule : BoundRule := .sum [.predecessor 0 167286 .coefficient, .predecessor 1 167287 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 167286 .coefficient)
      LeftBound167284.bound (LeftBound167284.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events653.exact167285RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound167284.bound, RecordedBoundRefines] <;> decide)
      (LeftBound167284.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 167287 .coefficient)
      LeftBound167265.bound (LeftBound167265.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events653.exact167270RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound167265.bound, RecordedBoundRefines] <;> decide)
      (LeftBound167265.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound167284.bound, LeftBound167265.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound167284.bound, LeftBound167265.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound167284.actual selector witness, LeftBound167265.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound167288

namespace LeftBound167301
def owner : Owner := ⟨.program ⟨257⟩, ⟨27965⟩⟩
def transferEvent : Nat := 167301
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 167299 .coefficient, .predecessor 1 167300 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 167299 .coefficient)
      LeftBound167122.bound (LeftBound167122.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events653.exact167298RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound167122.bound, RecordedBoundRefines] <;> decide)
      (LeftBound167122.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 167300 .coefficient)
      LeftBound167105.bound (LeftBound167105.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events652.exact167112RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound167105.bound, RecordedBoundRefines] <;> decide)
      (LeftBound167105.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound167122.bound, LeftBound167105.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound167122.bound, LeftBound167105.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound167122.actual selector witness, LeftBound167105.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound167301

namespace LeftBound167304
def owner : Owner := ⟨.program ⟨257⟩, ⟨27965⟩⟩
def transferEvent : Nat := 167304
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 167298 .summary, .result 167112 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 167298 .summary)
      LeftBound167124.bound (LeftBound167124.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨26892⟩⟩) (rawTerms := some (Proof.Events653.exact167298RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound167124.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 167112 .summary)
      LeftBound167107.bound (LeftBound167107.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨27964⟩⟩) (rawTerms := some (Proof.Events652.exact167112RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound167107.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound167124.bound, LeftBound167107.bound]
def bound : CoeffClass := .finite ⟨2998072422921948889088, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound167124.bound, LeftBound167107.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound167124.actual selector witness, LeftBound167107.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound167304

namespace LeftBound167308
def owner : Owner := ⟨.program ⟨257⟩, ⟨28391⟩⟩
def transferEvent : Nat := 167308
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 167306 .coefficient) (.predecessor 1 167307 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 167306 .coefficient)
      LeftBound167301.bound (LeftBound167301.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events653.exact167305RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound167301.bound, RecordedBoundRefines] <;> decide)
      (LeftBound167301.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 167307 .coefficient)
      LeftAuthority167027.bound (LeftAuthority167027.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events652.exact167028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority167027.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority167027.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound167301.bound LeftAuthority167027.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound167301.bound, LeftAuthority167027.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound167301.actual selector witness) * (LeftAuthority167027.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound167308

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
