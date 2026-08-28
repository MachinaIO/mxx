import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1939

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound286109
def owner : Owner := ⟨.program ⟨257⟩, ⟨56344⟩⟩
def transferEvent : Nat := 286109
def frameStart : Nat := 286080
def rule : BoundRule := .product (.predecessor 0 286107 .coefficient) (.predecessor 1 286108 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 286107 .coefficient)
      LeftAuthority286105.bound (LeftAuthority286105.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1117.exact286106RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority286105.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority286105.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 286108 .coefficient)
      LeftAuthority286102.bound (LeftAuthority286102.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1117.exact286103RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority286102.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority286102.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority286105.bound LeftAuthority286102.bound
def bound : CoeffClass := .finite ⟨256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority286105.bound, LeftAuthority286102.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority286105.actual selector witness) * (LeftAuthority286102.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound286109

namespace LeftBound286113
def owner : Owner := ⟨.program ⟨257⟩, ⟨56345⟩⟩
def transferEvent : Nat := 286113
def frameStart : Nat := 286080
def rule : BoundRule := .identity (.predecessor 0 286112 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 286112 .coefficient)
      LeftBound286109.bound (LeftBound286109.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1117.exact286111RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound286109.bound, RecordedBoundRefines] <;> decide)
      (LeftBound286109.derived selector witness)

def rawBound : CoeffClass := LeftBound286109.bound
def bound : CoeffClass := .finite ⟨256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound286109.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound286109.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound286113

namespace LeftBound286130
def owner : Owner := ⟨.program ⟨257⟩, ⟨58222⟩⟩
def transferEvent : Nat := 286130
def frameStart : Nat := 286080
def rule : BoundRule := .sum [.predecessor 0 286128 .coefficient, .predecessor 1 286129 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 286128 .coefficient)
      LeftBound286113.bound (LeftBound286113.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound286113.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 286129 .coefficient)
      LeftAuthority286126.bound (LeftAuthority286126.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority286126.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound286113.bound, LeftAuthority286126.bound]
def bound : CoeffClass := .finite ⟨256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound286113.bound, LeftAuthority286126.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound286113.actual selector witness, LeftAuthority286126.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound286130

namespace LeftBound286133
def owner : Owner := ⟨.program ⟨257⟩, ⟨58223⟩⟩
def transferEvent : Nat := 286133
def frameStart : Nat := 286080
def rule : BoundRule := .identity (.predecessor 0 286132 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 286132 .coefficient)
      LeftBound286130.bound (LeftBound286130.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound286130.derived selector witness)

def rawBound : CoeffClass := LeftBound286130.bound
def bound : CoeffClass := .finite ⟨256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound286130.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound286130.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound286133

namespace LeftBound286139
def owner : Owner := ⟨.program ⟨257⟩, ⟨58224⟩⟩
def transferEvent : Nat := 286139
def frameStart : Nat := 286080
def rule : BoundRule := .product (.predecessor 0 286137 .coefficient) (.predecessor 1 286138 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 286137 .coefficient)
      LeftAuthority286135.bound (LeftAuthority286135.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1117.exact286136RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority286135.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority286135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 286138 .coefficient)
      LeftBound286133.bound (LeftBound286133.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1117.exact286134RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound286133.bound, RecordedBoundRefines] <;> decide)
      (LeftBound286133.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority286135.bound LeftBound286133.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority286135.bound, LeftBound286133.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority286135.actual selector witness) * (LeftBound286133.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound286139

namespace LeftBound286153
def owner : Owner := ⟨.program ⟨257⟩, ⟨9533⟩⟩
def transferEvent : Nat := 286153
def frameStart : Nat := 286080
def rule : BoundRule := .scale (.predecessor 0 286151 .coefficient) (.value (.predecessor 1 286152 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 286151 .coefficient)
      LeftAuthority286149.bound (LeftAuthority286149.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1117.exact286150RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority286149.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority286149.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 286152 .coefficient)
      LeftAuthority286083.bound (LeftAuthority286083.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority286083.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority286149.bound LeftAuthority286083.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority286149.bound, LeftAuthority286083.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority286149.actual selector witness) * (LeftAuthority286083.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound286153

namespace LeftBound286156
def owner : Owner := ⟨.program ⟨257⟩, ⟨7290⟩⟩
def transferEvent : Nat := 286156
def frameStart : Nat := 286080
def rule : BoundRule := .identity (.predecessor 0 286155 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 286155 .coefficient)
      LeftAuthority286143.bound (LeftAuthority286143.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1117.exact286144RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority286143.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority286143.derived selector witness)

def rawBound : CoeffClass := LeftAuthority286143.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority286143.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority286143.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound286156

namespace LeftBound286160
def owner : Owner := ⟨.program ⟨257⟩, ⟨9534⟩⟩
def transferEvent : Nat := 286160
def frameStart : Nat := 286080
def rule : BoundRule := .product (.predecessor 0 286158 .coefficient) (.predecessor 1 286159 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 286158 .coefficient)
      LeftBound286156.bound (LeftBound286156.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1117.exact286157RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound286156.bound, RecordedBoundRefines] <;> decide)
      (LeftBound286156.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 286159 .coefficient)
      LeftBound286153.bound (LeftBound286153.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1117.exact286154RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound286153.bound, RecordedBoundRefines] <;> decide)
      (LeftBound286153.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound286156.bound LeftBound286153.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound286156.bound, LeftBound286153.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound286156.actual selector witness) * (LeftBound286153.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound286160

namespace LeftBound286165
def owner : Owner := ⟨.program ⟨257⟩, ⟨58225⟩⟩
def transferEvent : Nat := 286165
def frameStart : Nat := 286080
def rule : BoundRule := .sum [.predecessor 0 286163 .coefficient, .predecessor 1 286164 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 286163 .coefficient)
      LeftBound286160.bound (LeftBound286160.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1117.exact286162RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound286160.bound, RecordedBoundRefines] <;> decide)
      (LeftBound286160.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 286164 .coefficient)
      LeftBound286139.bound (LeftBound286139.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1117.exact286141RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound286139.bound, RecordedBoundRefines] <;> decide)
      (LeftBound286139.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound286160.bound, LeftBound286139.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound286160.bound, LeftBound286139.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound286160.actual selector witness, LeftBound286139.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound286165

namespace LeftBound286169
def owner : Owner := ⟨.program ⟨257⟩, ⟨58416⟩⟩
def transferEvent : Nat := 286169
def frameStart : Nat := 286080
def rule : BoundRule := .product (.predecessor 0 286167 .coefficient) (.predecessor 1 286168 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 286167 .coefficient)
      LeftBound286165.bound (LeftBound286165.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1117.exact286166RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound286165.bound, RecordedBoundRefines] <;> decide)
      (LeftBound286165.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 286168 .coefficient)
      LeftAuthority286124.bound (LeftAuthority286124.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1117.exact286125RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority286124.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority286124.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound286165.bound LeftAuthority286124.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound286165.bound, LeftAuthority286124.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound286165.actual selector witness) * (LeftAuthority286124.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound286169

namespace LeftBound286180
def owner : Owner := ⟨.program ⟨257⟩, ⟨56802⟩⟩
def transferEvent : Nat := 286180
def frameStart : Nat := 286080
def rule : BoundRule := .product (.predecessor 0 286178 .coefficient) (.predecessor 1 286179 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 286178 .coefficient)
      LeftAuthority286135.bound (LeftAuthority286135.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1117.exact286136RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority286135.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority286135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 286179 .coefficient)
      LeftAuthority286176.bound (LeftAuthority286176.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1117.exact286177RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority286176.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority286176.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority286135.bound LeftAuthority286176.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority286135.bound, LeftAuthority286176.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority286135.actual selector witness) * (LeftAuthority286176.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound286180

namespace LeftBound286188
def owner : Owner := ⟨.program ⟨257⟩, ⟨56803⟩⟩
def transferEvent : Nat := 286188
def frameStart : Nat := 286080
def rule : BoundRule := .sum [.predecessor 0 286186 .coefficient, .predecessor 1 286187 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 286186 .coefficient)
      LeftAuthority286184.bound (LeftAuthority286184.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1117.exact286185RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority286184.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority286184.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 286187 .coefficient)
      LeftBound286180.bound (LeftBound286180.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1117.exact286182RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound286180.bound, RecordedBoundRefines] <;> decide)
      (LeftBound286180.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority286184.bound, LeftBound286180.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority286184.bound, LeftBound286180.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority286184.actual selector witness, LeftBound286180.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound286188

namespace LeftBound286192
def owner : Owner := ⟨.program ⟨257⟩, ⟨58417⟩⟩
def transferEvent : Nat := 286192
def frameStart : Nat := 286080
def rule : BoundRule := .sum [.predecessor 0 286190 .coefficient, .predecessor 1 286191 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 286190 .coefficient)
      LeftBound286188.bound (LeftBound286188.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1117.exact286189RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound286188.bound, RecordedBoundRefines] <;> decide)
      (LeftBound286188.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 286191 .coefficient)
      LeftBound286169.bound (LeftBound286169.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1117.exact286174RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound286169.bound, RecordedBoundRefines] <;> decide)
      (LeftBound286169.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound286188.bound, LeftBound286169.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound286188.bound, LeftBound286169.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound286188.actual selector witness, LeftBound286169.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound286192

namespace LeftBound286205
def owner : Owner := ⟨.program ⟨257⟩, ⟨58415⟩⟩
def transferEvent : Nat := 286205
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 286203 .coefficient, .predecessor 1 286204 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 286203 .coefficient)
      LeftBound286028.bound (LeftBound286028.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1117.exact286202RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound286028.bound, RecordedBoundRefines] <;> decide)
      (LeftBound286028.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 286204 .coefficient)
      LeftBound286011.bound (LeftBound286011.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1117.exact286018RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound286011.bound, RecordedBoundRefines] <;> decide)
      (LeftBound286011.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound286028.bound, LeftBound286011.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound286028.bound, LeftBound286011.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound286028.actual selector witness, LeftBound286011.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound286205

namespace LeftBound286208
def owner : Owner := ⟨.program ⟨257⟩, ⟨58415⟩⟩
def transferEvent : Nat := 286208
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 286202 .summary, .result 286018 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 286202 .summary)
      LeftBound286030.bound (LeftBound286030.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨57352⟩⟩) (rawTerms := some (Proof.Events1117.exact286202RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound286030.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 286018 .summary)
      LeftBound286013.bound (LeftBound286013.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58414⟩⟩) (rawTerms := some (Proof.Events1117.exact286018RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound286013.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound286030.bound, LeftBound286013.bound]
def bound : CoeffClass := .finite ⟨2997944351807545540608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound286030.bound, LeftBound286013.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound286030.actual selector witness, LeftBound286013.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound286208

namespace LeftBound286212
def owner : Owner := ⟨.program ⟨257⟩, ⟨58728⟩⟩
def transferEvent : Nat := 286212
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 286210 .coefficient) (.predecessor 1 286211 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 286210 .coefficient)
      LeftBound286205.bound (LeftBound286205.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1118.exact286209RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound286205.bound, RecordedBoundRefines] <;> decide)
      (LeftBound286205.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 286211 .coefficient)
      LeftAuthority285933.bound (LeftAuthority285933.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1116.exact285934RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority285933.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority285933.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound286205.bound LeftAuthority285933.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound286205.bound, LeftAuthority285933.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound286205.actual selector witness) * (LeftAuthority285933.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound286212

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
