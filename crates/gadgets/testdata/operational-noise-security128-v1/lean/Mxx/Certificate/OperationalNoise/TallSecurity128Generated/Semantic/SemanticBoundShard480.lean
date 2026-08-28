import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard479

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound76079
def owner : Owner := ⟨.program ⟨257⟩, ⟨47979⟩⟩
def transferEvent : Nat := 76079
def frameStart : Nat := 76050
def rule : BoundRule := .product (.predecessor 0 76077 .coefficient) (.predecessor 1 76078 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 76077 .coefficient)
      LeftAuthority76075.bound (LeftAuthority76075.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events297.exact76076RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76075.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76075.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 76078 .coefficient)
      LeftAuthority76072.bound (LeftAuthority76072.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events297.exact76073RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76072.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76072.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority76075.bound LeftAuthority76072.bound
def bound : CoeffClass := .finite ⟨3600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority76075.bound, LeftAuthority76072.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority76075.actual selector witness) * (LeftAuthority76072.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76079

namespace LeftBound76083
def owner : Owner := ⟨.program ⟨257⟩, ⟨47980⟩⟩
def transferEvent : Nat := 76083
def frameStart : Nat := 76050
def rule : BoundRule := .identity (.predecessor 0 76082 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 76082 .coefficient)
      LeftBound76079.bound (LeftBound76079.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events297.exact76081RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76079.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76079.derived selector witness)

def rawBound : CoeffClass := LeftBound76079.bound
def bound : CoeffClass := .finite ⟨3600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76079.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound76079.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound76083

namespace LeftBound76100
def owner : Owner := ⟨.program ⟨257⟩, ⟨49450⟩⟩
def transferEvent : Nat := 76100
def frameStart : Nat := 76050
def rule : BoundRule := .sum [.predecessor 0 76098 .coefficient, .predecessor 1 76099 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 76098 .coefficient)
      LeftBound76083.bound (LeftBound76083.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound76083.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 76099 .coefficient)
      LeftAuthority76096.bound (LeftAuthority76096.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority76096.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound76083.bound, LeftAuthority76096.bound]
def bound : CoeffClass := .finite ⟨3600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76083.bound, LeftAuthority76096.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound76083.actual selector witness, LeftAuthority76096.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound76100

namespace LeftBound76103
def owner : Owner := ⟨.program ⟨257⟩, ⟨49451⟩⟩
def transferEvent : Nat := 76103
def frameStart : Nat := 76050
def rule : BoundRule := .identity (.predecessor 0 76102 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 76102 .coefficient)
      LeftBound76100.bound (LeftBound76100.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound76100.derived selector witness)

def rawBound : CoeffClass := LeftBound76100.bound
def bound : CoeffClass := .finite ⟨3600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76100.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound76100.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound76103

namespace LeftBound76109
def owner : Owner := ⟨.program ⟨257⟩, ⟨49452⟩⟩
def transferEvent : Nat := 76109
def frameStart : Nat := 76050
def rule : BoundRule := .product (.predecessor 0 76107 .coefficient) (.predecessor 1 76108 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 76107 .coefficient)
      LeftAuthority76105.bound (LeftAuthority76105.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events297.exact76106RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76105.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76105.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 76108 .coefficient)
      LeftBound76103.bound (LeftBound76103.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events297.exact76104RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76103.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76103.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority76105.bound LeftBound76103.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority76105.bound, LeftBound76103.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority76105.actual selector witness) * (LeftBound76103.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76109

namespace LeftBound76125
def owner : Owner := ⟨.program ⟨257⟩, ⟨9566⟩⟩
def transferEvent : Nat := 76125
def frameStart : Nat := 76050
def rule : BoundRule := .scale (.predecessor 0 76123 .coefficient) (.value (.predecessor 1 76124 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 76123 .coefficient)
      LeftAuthority76121.bound (LeftAuthority76121.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events297.exact76122RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76121.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76121.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 76124 .coefficient)
      LeftAuthority76112.bound (LeftAuthority76112.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority76112.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority76121.bound LeftAuthority76112.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority76121.bound, LeftAuthority76112.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority76121.actual selector witness) * (LeftAuthority76112.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound76125

namespace LeftBound76128
def owner : Owner := ⟨.program ⟨257⟩, ⟨7302⟩⟩
def transferEvent : Nat := 76128
def frameStart : Nat := 76050
def rule : BoundRule := .identity (.predecessor 0 76127 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 76127 .coefficient)
      LeftAuthority76115.bound (LeftAuthority76115.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events297.exact76116RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76115.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76115.derived selector witness)

def rawBound : CoeffClass := LeftAuthority76115.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority76115.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority76115.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound76128

namespace LeftBound76132
def owner : Owner := ⟨.program ⟨257⟩, ⟨9567⟩⟩
def transferEvent : Nat := 76132
def frameStart : Nat := 76050
def rule : BoundRule := .product (.predecessor 0 76130 .coefficient) (.predecessor 1 76131 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 76130 .coefficient)
      LeftBound76128.bound (LeftBound76128.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events297.exact76129RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76128.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76128.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 76131 .coefficient)
      LeftBound76125.bound (LeftBound76125.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events297.exact76126RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76125.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76125.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound76128.bound LeftBound76125.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76128.bound, LeftBound76125.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound76128.actual selector witness) * (LeftBound76125.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76132

namespace LeftBound76137
def owner : Owner := ⟨.program ⟨257⟩, ⟨49453⟩⟩
def transferEvent : Nat := 76137
def frameStart : Nat := 76050
def rule : BoundRule := .sum [.predecessor 0 76135 .coefficient, .predecessor 1 76136 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 76135 .coefficient)
      LeftBound76132.bound (LeftBound76132.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events297.exact76134RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76132.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76132.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 76136 .coefficient)
      LeftBound76109.bound (LeftBound76109.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events297.exact76111RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76109.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76109.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound76132.bound, LeftBound76109.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76132.bound, LeftBound76109.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound76132.actual selector witness, LeftBound76109.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound76137

namespace LeftBound76141
def owner : Owner := ⟨.program ⟨257⟩, ⟨49728⟩⟩
def transferEvent : Nat := 76141
def frameStart : Nat := 76050
def rule : BoundRule := .product (.predecessor 0 76139 .coefficient) (.predecessor 1 76140 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 76139 .coefficient)
      LeftBound76137.bound (LeftBound76137.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events297.exact76138RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76137.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76137.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 76140 .coefficient)
      LeftAuthority76094.bound (LeftAuthority76094.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events297.exact76095RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76094.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76094.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound76137.bound LeftAuthority76094.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76137.bound, LeftAuthority76094.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound76137.actual selector witness) * (LeftAuthority76094.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76141

namespace LeftBound76152
def owner : Owner := ⟨.program ⟨257⟩, ⟨48198⟩⟩
def transferEvent : Nat := 76152
def frameStart : Nat := 76050
def rule : BoundRule := .product (.predecessor 0 76150 .coefficient) (.predecessor 1 76151 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 76150 .coefficient)
      LeftAuthority76105.bound (LeftAuthority76105.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events297.exact76106RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76105.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76105.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 76151 .coefficient)
      LeftAuthority76148.bound (LeftAuthority76148.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events297.exact76149RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76148.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76148.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority76105.bound LeftAuthority76148.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority76105.bound, LeftAuthority76148.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority76105.actual selector witness) * (LeftAuthority76148.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76152

namespace LeftBound76160
def owner : Owner := ⟨.program ⟨257⟩, ⟨48199⟩⟩
def transferEvent : Nat := 76160
def frameStart : Nat := 76050
def rule : BoundRule := .sum [.predecessor 0 76158 .coefficient, .predecessor 1 76159 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 76158 .coefficient)
      LeftAuthority76156.bound (LeftAuthority76156.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events297.exact76157RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76156.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76156.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 76159 .coefficient)
      LeftBound76152.bound (LeftBound76152.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events297.exact76154RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76152.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76152.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority76156.bound, LeftBound76152.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority76156.bound, LeftBound76152.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority76156.actual selector witness, LeftBound76152.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound76160

namespace LeftBound76164
def owner : Owner := ⟨.program ⟨257⟩, ⟨49729⟩⟩
def transferEvent : Nat := 76164
def frameStart : Nat := 76050
def rule : BoundRule := .sum [.predecessor 0 76162 .coefficient, .predecessor 1 76163 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 76162 .coefficient)
      LeftBound76160.bound (LeftBound76160.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events297.exact76161RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76160.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76160.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 76163 .coefficient)
      LeftBound76141.bound (LeftBound76141.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events297.exact76146RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76141.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76141.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound76160.bound, LeftBound76141.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76160.bound, LeftBound76141.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound76160.actual selector witness, LeftBound76141.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound76164

namespace LeftBound76177
def owner : Owner := ⟨.program ⟨257⟩, ⟨49727⟩⟩
def transferEvent : Nat := 76177
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 76175 .coefficient, .predecessor 1 76176 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 76175 .coefficient)
      LeftBound75998.bound (LeftBound75998.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events297.exact76174RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75998.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75998.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 76176 .coefficient)
      LeftBound75970.bound (LeftBound75970.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events296.exact75977RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75970.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75970.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75998.bound, LeftBound75970.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75998.bound, LeftBound75970.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound75998.actual selector witness, LeftBound75970.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound76177

namespace LeftBound76180
def owner : Owner := ⟨.program ⟨257⟩, ⟨49727⟩⟩
def transferEvent : Nat := 76180
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 76174 .summary, .result 75977 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 76174 .summary)
      LeftBound76000.bound (LeftBound76000.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨48652⟩⟩) (rawTerms := some (Proof.Events297.exact76174RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound76000.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 75977 .summary)
      LeftBound75972.bound (LeftBound75972.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨49726⟩⟩) (rawTerms := some (Proof.Events296.exact75977RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound75972.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound76000.bound, LeftBound75972.bound]
def bound : CoeffClass := .finite ⟨2998346861024241778688, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76000.bound, LeftBound75972.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound76000.actual selector witness, LeftBound75972.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound76180

namespace LeftBound76184
def owner : Owner := ⟨.program ⟨257⟩, ⟨50181⟩⟩
def transferEvent : Nat := 76184
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 76182 .coefficient) (.predecessor 1 76183 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 76182 .coefficient)
      LeftBound76177.bound (LeftBound76177.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events297.exact76181RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76177.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76177.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 76183 .coefficient)
      LeftAuthority75887.bound (LeftAuthority75887.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events296.exact75888RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75887.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75887.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound76177.bound LeftAuthority75887.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76177.bound, LeftAuthority75887.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound76177.actual selector witness) * (LeftAuthority75887.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76184

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
