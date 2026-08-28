import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1155

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound173607
def owner : Owner := ⟨.program ⟨257⟩, ⟨69103⟩⟩
def transferEvent : Nat := 173607
def frameStart : Nat := 173086
def rule : BoundRule := .sum [.predecessor 0 173605 .coefficient, .predecessor 1 173606 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 173605 .coefficient)
      LeftBound173590.bound (LeftBound173590.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound173590.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 173606 .coefficient)
      LeftAuthority173603.bound (LeftAuthority173603.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority173603.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound173590.bound, LeftAuthority173603.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound173590.bound, LeftAuthority173603.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound173590.actual selector witness, LeftAuthority173603.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound173607

namespace LeftBound173610
def owner : Owner := ⟨.program ⟨257⟩, ⟨69104⟩⟩
def transferEvent : Nat := 173610
def frameStart : Nat := 173086
def rule : BoundRule := .identity (.predecessor 0 173609 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 173609 .coefficient)
      LeftBound173607.bound (LeftBound173607.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound173607.derived selector witness)

def rawBound : CoeffClass := LeftBound173607.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound173607.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound173607.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound173610

namespace LeftBound173616
def owner : Owner := ⟨.program ⟨257⟩, ⟨69105⟩⟩
def transferEvent : Nat := 173616
def frameStart : Nat := 173086
def rule : BoundRule := .product (.predecessor 0 173614 .coefficient) (.predecessor 1 173615 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 173614 .coefficient)
      LeftAuthority173612.bound (LeftAuthority173612.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events678.exact173613RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority173612.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority173612.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 173615 .coefficient)
      LeftBound173610.bound (LeftBound173610.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events678.exact173611RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound173610.bound, RecordedBoundRefines] <;> decide)
      (LeftBound173610.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority173612.bound LeftBound173610.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority173612.bound, LeftBound173610.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority173612.actual selector witness) * (LeftBound173610.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound173616

namespace LeftBound173692
def owner : Owner := ⟨.program ⟨257⟩, ⟨7309⟩⟩
def transferEvent : Nat := 173692
def frameStart : Nat := 173086
def rule : BoundRule := .sum [.predecessor 0 173690 .coefficient, .predecessor 1 173691 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 173690 .coefficient)
      LeftAuthority173688.bound (LeftAuthority173688.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events678.exact173689RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority173688.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority173688.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 173691 .coefficient)
      LeftAuthority173685.bound (LeftAuthority173685.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events678.exact173686RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority173685.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority173685.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority173688.bound, LeftAuthority173685.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority173688.bound, LeftAuthority173685.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority173688.actual selector witness, LeftAuthority173685.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound173692

namespace LeftBound173696
def owner : Owner := ⟨.program ⟨257⟩, ⟨7310⟩⟩
def transferEvent : Nat := 173696
def frameStart : Nat := 173086
def rule : BoundRule := .sum [.predecessor 0 173694 .coefficient, .predecessor 1 173695 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 173694 .coefficient)
      LeftBound173692.bound (LeftBound173692.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events678.exact173693RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound173692.bound, RecordedBoundRefines] <;> decide)
      (LeftBound173692.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 173695 .coefficient)
      LeftAuthority173682.bound (LeftAuthority173682.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events678.exact173683RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority173682.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority173682.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound173692.bound, LeftAuthority173682.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound173692.bound, LeftAuthority173682.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound173692.actual selector witness, LeftAuthority173682.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound173696

namespace LeftBound173700
def owner : Owner := ⟨.program ⟨257⟩, ⟨7311⟩⟩
def transferEvent : Nat := 173700
def frameStart : Nat := 173086
def rule : BoundRule := .sum [.predecessor 0 173698 .coefficient, .predecessor 1 173699 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 173698 .coefficient)
      LeftBound173696.bound (LeftBound173696.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events678.exact173697RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound173696.bound, RecordedBoundRefines] <;> decide)
      (LeftBound173696.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 173699 .coefficient)
      LeftAuthority173679.bound (LeftAuthority173679.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events678.exact173680RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority173679.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority173679.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound173696.bound, LeftAuthority173679.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound173696.bound, LeftAuthority173679.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound173696.actual selector witness, LeftAuthority173679.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound173700

namespace LeftBound173704
def owner : Owner := ⟨.program ⟨257⟩, ⟨7312⟩⟩
def transferEvent : Nat := 173704
def frameStart : Nat := 173086
def rule : BoundRule := .sum [.predecessor 0 173702 .coefficient, .predecessor 1 173703 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 173702 .coefficient)
      LeftBound173700.bound (LeftBound173700.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events678.exact173701RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound173700.bound, RecordedBoundRefines] <;> decide)
      (LeftBound173700.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 173703 .coefficient)
      LeftAuthority173676.bound (LeftAuthority173676.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events678.exact173677RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority173676.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority173676.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound173700.bound, LeftAuthority173676.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound173700.bound, LeftAuthority173676.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound173700.actual selector witness, LeftAuthority173676.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound173704

namespace LeftBound173708
def owner : Owner := ⟨.program ⟨257⟩, ⟨7313⟩⟩
def transferEvent : Nat := 173708
def frameStart : Nat := 173086
def rule : BoundRule := .sum [.predecessor 0 173706 .coefficient, .predecessor 1 173707 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 173706 .coefficient)
      LeftBound173704.bound (LeftBound173704.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events678.exact173705RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound173704.bound, RecordedBoundRefines] <;> decide)
      (LeftBound173704.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 173707 .coefficient)
      LeftAuthority173673.bound (LeftAuthority173673.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events678.exact173674RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority173673.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority173673.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound173704.bound, LeftAuthority173673.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound173704.bound, LeftAuthority173673.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound173704.actual selector witness, LeftAuthority173673.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound173708

namespace LeftBound173712
def owner : Owner := ⟨.program ⟨257⟩, ⟨7314⟩⟩
def transferEvent : Nat := 173712
def frameStart : Nat := 173086
def rule : BoundRule := .sum [.predecessor 0 173710 .coefficient, .predecessor 1 173711 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 173710 .coefficient)
      LeftBound173708.bound (LeftBound173708.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events678.exact173709RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound173708.bound, RecordedBoundRefines] <;> decide)
      (LeftBound173708.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 173711 .coefficient)
      LeftAuthority173670.bound (LeftAuthority173670.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events678.exact173671RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority173670.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority173670.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound173708.bound, LeftAuthority173670.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound173708.bound, LeftAuthority173670.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound173708.actual selector witness, LeftAuthority173670.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound173712

namespace LeftBound173716
def owner : Owner := ⟨.program ⟨257⟩, ⟨7315⟩⟩
def transferEvent : Nat := 173716
def frameStart : Nat := 173086
def rule : BoundRule := .sum [.predecessor 0 173714 .coefficient, .predecessor 1 173715 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 173714 .coefficient)
      LeftBound173712.bound (LeftBound173712.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events678.exact173713RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound173712.bound, RecordedBoundRefines] <;> decide)
      (LeftBound173712.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 173715 .coefficient)
      LeftAuthority173667.bound (LeftAuthority173667.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events678.exact173668RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority173667.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority173667.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound173712.bound, LeftAuthority173667.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound173712.bound, LeftAuthority173667.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound173712.actual selector witness, LeftAuthority173667.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound173716

namespace LeftBound173720
def owner : Owner := ⟨.program ⟨257⟩, ⟨7316⟩⟩
def transferEvent : Nat := 173720
def frameStart : Nat := 173086
def rule : BoundRule := .sum [.predecessor 0 173718 .coefficient, .predecessor 1 173719 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 173718 .coefficient)
      LeftBound173716.bound (LeftBound173716.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events678.exact173717RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound173716.bound, RecordedBoundRefines] <;> decide)
      (LeftBound173716.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 173719 .coefficient)
      LeftAuthority173664.bound (LeftAuthority173664.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events678.exact173665RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority173664.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority173664.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound173716.bound, LeftAuthority173664.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound173716.bound, LeftAuthority173664.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound173716.actual selector witness, LeftAuthority173664.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound173720

namespace LeftBound173724
def owner : Owner := ⟨.program ⟨257⟩, ⟨7317⟩⟩
def transferEvent : Nat := 173724
def frameStart : Nat := 173086
def rule : BoundRule := .sum [.predecessor 0 173722 .coefficient, .predecessor 1 173723 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 173722 .coefficient)
      LeftBound173720.bound (LeftBound173720.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events678.exact173721RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound173720.bound, RecordedBoundRefines] <;> decide)
      (LeftBound173720.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 173723 .coefficient)
      LeftAuthority173661.bound (LeftAuthority173661.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events678.exact173662RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority173661.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority173661.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound173720.bound, LeftAuthority173661.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound173720.bound, LeftAuthority173661.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound173720.actual selector witness, LeftAuthority173661.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound173724

namespace LeftBound173728
def owner : Owner := ⟨.program ⟨257⟩, ⟨7318⟩⟩
def transferEvent : Nat := 173728
def frameStart : Nat := 173086
def rule : BoundRule := .sum [.predecessor 0 173726 .coefficient, .predecessor 1 173727 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 173726 .coefficient)
      LeftBound173724.bound (LeftBound173724.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events678.exact173725RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound173724.bound, RecordedBoundRefines] <;> decide)
      (LeftBound173724.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 173727 .coefficient)
      LeftAuthority173658.bound (LeftAuthority173658.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events678.exact173659RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority173658.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority173658.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound173724.bound, LeftAuthority173658.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound173724.bound, LeftAuthority173658.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound173724.actual selector witness, LeftAuthority173658.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound173728

namespace LeftBound173732
def owner : Owner := ⟨.program ⟨257⟩, ⟨7319⟩⟩
def transferEvent : Nat := 173732
def frameStart : Nat := 173086
def rule : BoundRule := .sum [.predecessor 0 173730 .coefficient, .predecessor 1 173731 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 173730 .coefficient)
      LeftBound173728.bound (LeftBound173728.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events678.exact173729RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound173728.bound, RecordedBoundRefines] <;> decide)
      (LeftBound173728.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 173731 .coefficient)
      LeftAuthority173655.bound (LeftAuthority173655.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events678.exact173656RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority173655.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority173655.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound173728.bound, LeftAuthority173655.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound173728.bound, LeftAuthority173655.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound173728.actual selector witness, LeftAuthority173655.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound173732

namespace LeftBound173736
def owner : Owner := ⟨.program ⟨257⟩, ⟨7320⟩⟩
def transferEvent : Nat := 173736
def frameStart : Nat := 173086
def rule : BoundRule := .sum [.predecessor 0 173734 .coefficient, .predecessor 1 173735 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 173734 .coefficient)
      LeftBound173732.bound (LeftBound173732.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events678.exact173733RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound173732.bound, RecordedBoundRefines] <;> decide)
      (LeftBound173732.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 173735 .coefficient)
      LeftAuthority173652.bound (LeftAuthority173652.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events678.exact173653RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority173652.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority173652.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound173732.bound, LeftAuthority173652.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound173732.bound, LeftAuthority173652.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound173732.actual selector witness, LeftAuthority173652.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound173736

namespace LeftBound173740
def owner : Owner := ⟨.program ⟨257⟩, ⟨7321⟩⟩
def transferEvent : Nat := 173740
def frameStart : Nat := 173086
def rule : BoundRule := .sum [.predecessor 0 173738 .coefficient, .predecessor 1 173739 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 173738 .coefficient)
      LeftBound173736.bound (LeftBound173736.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events678.exact173737RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound173736.bound, RecordedBoundRefines] <;> decide)
      (LeftBound173736.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 173739 .coefficient)
      LeftAuthority173649.bound (LeftAuthority173649.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events678.exact173650RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority173649.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority173649.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound173736.bound, LeftAuthority173649.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound173736.bound, LeftAuthority173649.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound173736.actual selector witness, LeftAuthority173649.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound173740

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
