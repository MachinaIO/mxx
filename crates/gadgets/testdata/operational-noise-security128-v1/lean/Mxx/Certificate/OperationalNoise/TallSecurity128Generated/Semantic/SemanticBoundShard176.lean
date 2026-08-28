import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard175

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound32250
def owner : Owner := ⟨.program ⟨257⟩, ⟨9566⟩⟩
def transferEvent : Nat := 32250
def frameStart : Nat := 32175
def rule : BoundRule := .scale (.predecessor 0 32248 .coefficient) (.value (.predecessor 1 32249 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 32248 .coefficient)
      LeftAuthority32246.bound (LeftAuthority32246.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events125.exact32247RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32246.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32246.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 32249 .coefficient)
      LeftAuthority32237.bound (LeftAuthority32237.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority32237.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority32246.bound LeftAuthority32237.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority32246.bound, LeftAuthority32237.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority32246.actual selector witness) * (LeftAuthority32237.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound32250

namespace LeftBound32253
def owner : Owner := ⟨.program ⟨257⟩, ⟨7302⟩⟩
def transferEvent : Nat := 32253
def frameStart : Nat := 32175
def rule : BoundRule := .identity (.predecessor 0 32252 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 32252 .coefficient)
      LeftAuthority32240.bound (LeftAuthority32240.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events125.exact32241RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32240.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32240.derived selector witness)

def rawBound : CoeffClass := LeftAuthority32240.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority32240.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority32240.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound32253

namespace LeftBound32257
def owner : Owner := ⟨.program ⟨257⟩, ⟨9567⟩⟩
def transferEvent : Nat := 32257
def frameStart : Nat := 32175
def rule : BoundRule := .product (.predecessor 0 32255 .coefficient) (.predecessor 1 32256 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 32255 .coefficient)
      LeftBound32253.bound (LeftBound32253.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events125.exact32254RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32253.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32253.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 32256 .coefficient)
      LeftBound32250.bound (LeftBound32250.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events125.exact32251RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32250.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32250.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound32253.bound LeftBound32250.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32253.bound, LeftBound32250.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound32253.actual selector witness) * (LeftBound32250.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound32257

namespace LeftBound32262
def owner : Owner := ⟨.program ⟨257⟩, ⟨49465⟩⟩
def transferEvent : Nat := 32262
def frameStart : Nat := 32175
def rule : BoundRule := .sum [.predecessor 0 32260 .coefficient, .predecessor 1 32261 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 32260 .coefficient)
      LeftBound32257.bound (LeftBound32257.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events126.exact32259RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32257.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32257.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 32261 .coefficient)
      LeftBound32234.bound (LeftBound32234.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events125.exact32236RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32234.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32234.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound32257.bound, LeftBound32234.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32257.bound, LeftBound32234.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound32257.actual selector witness, LeftBound32234.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound32262

namespace LeftBound32266
def owner : Owner := ⟨.program ⟨257⟩, ⟨49761⟩⟩
def transferEvent : Nat := 32266
def frameStart : Nat := 32175
def rule : BoundRule := .product (.predecessor 0 32264 .coefficient) (.predecessor 1 32265 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 32264 .coefficient)
      LeftBound32262.bound (LeftBound32262.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events126.exact32263RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32262.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32262.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 32265 .coefficient)
      LeftAuthority32219.bound (LeftAuthority32219.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events125.exact32220RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32219.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32219.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound32262.bound LeftAuthority32219.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32262.bound, LeftAuthority32219.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound32262.actual selector witness) * (LeftAuthority32219.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound32266

namespace LeftBound32277
def owner : Owner := ⟨.program ⟨257⟩, ⟨48222⟩⟩
def transferEvent : Nat := 32277
def frameStart : Nat := 32175
def rule : BoundRule := .product (.predecessor 0 32275 .coefficient) (.predecessor 1 32276 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 32275 .coefficient)
      LeftAuthority32230.bound (LeftAuthority32230.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events125.exact32231RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32230.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32230.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 32276 .coefficient)
      LeftAuthority32273.bound (LeftAuthority32273.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events126.exact32274RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32273.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32273.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority32230.bound LeftAuthority32273.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority32230.bound, LeftAuthority32273.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority32230.actual selector witness) * (LeftAuthority32273.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound32277

namespace LeftBound32285
def owner : Owner := ⟨.program ⟨257⟩, ⟨48223⟩⟩
def transferEvent : Nat := 32285
def frameStart : Nat := 32175
def rule : BoundRule := .sum [.predecessor 0 32283 .coefficient, .predecessor 1 32284 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 32283 .coefficient)
      LeftAuthority32281.bound (LeftAuthority32281.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events126.exact32282RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32281.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32281.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 32284 .coefficient)
      LeftBound32277.bound (LeftBound32277.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events126.exact32279RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32277.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32277.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority32281.bound, LeftBound32277.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority32281.bound, LeftBound32277.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority32281.actual selector witness, LeftBound32277.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound32285

namespace LeftBound32289
def owner : Owner := ⟨.program ⟨257⟩, ⟨49762⟩⟩
def transferEvent : Nat := 32289
def frameStart : Nat := 32175
def rule : BoundRule := .sum [.predecessor 0 32287 .coefficient, .predecessor 1 32288 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 32287 .coefficient)
      LeftBound32285.bound (LeftBound32285.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events126.exact32286RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32285.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32285.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 32288 .coefficient)
      LeftBound32266.bound (LeftBound32266.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events126.exact32271RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32266.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32266.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound32285.bound, LeftBound32266.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32285.bound, LeftBound32266.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound32285.actual selector witness, LeftBound32266.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound32289

namespace LeftBound32302
def owner : Owner := ⟨.program ⟨257⟩, ⟨49760⟩⟩
def transferEvent : Nat := 32302
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 32300 .coefficient, .predecessor 1 32301 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 32300 .coefficient)
      LeftBound32123.bound (LeftBound32123.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events126.exact32299RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32123.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32123.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 32301 .coefficient)
      LeftBound32095.bound (LeftBound32095.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events125.exact32102RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32095.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32095.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound32123.bound, LeftBound32095.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32123.bound, LeftBound32095.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound32123.actual selector witness, LeftBound32095.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound32302

namespace LeftBound32305
def owner : Owner := ⟨.program ⟨257⟩, ⟨49760⟩⟩
def transferEvent : Nat := 32305
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 32299 .summary, .result 32102 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 32299 .summary)
      LeftBound32125.bound (LeftBound32125.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨48682⟩⟩) (rawTerms := some (Proof.Events126.exact32299RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound32125.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 32102 .summary)
      LeftBound32097.bound (LeftBound32097.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨49759⟩⟩) (rawTerms := some (Proof.Events125.exact32102RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound32097.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound32125.bound, LeftBound32097.bound]
def bound : CoeffClass := .finite ⟨2998346861024241778688, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32125.bound, LeftBound32097.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound32125.actual selector witness, LeftBound32097.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound32305

namespace LeftBound32309
def owner : Owner := ⟨.program ⟨257⟩, ⟨50256⟩⟩
def transferEvent : Nat := 32309
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 32307 .coefficient) (.predecessor 1 32308 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 32307 .coefficient)
      LeftBound32302.bound (LeftBound32302.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events126.exact32306RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32302.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32302.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 32308 .coefficient)
      LeftAuthority32012.bound (LeftAuthority32012.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events125.exact32013RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32012.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32012.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound32302.bound LeftAuthority32012.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32302.bound, LeftAuthority32012.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound32302.actual selector witness) * (LeftAuthority32012.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound32309

namespace LeftBound32310
def owner : Owner := ⟨.program ⟨257⟩, ⟨50256⟩⟩
def transferEvent : Nat := 32310
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨50254⟩⟩]⟩ [⟨.result 32013 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 32013 .coefficient)
      LeftAuthority32012.bound (LeftAuthority32012.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨50254⟩⟩) (rawTerms := some (Proof.Events125.exact32013RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32012.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32012.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority32012.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority32012.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority32012.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound32310

namespace LeftBound32311
def owner : Owner := ⟨.program ⟨257⟩, ⟨50256⟩⟩
def transferEvent : Nat := 32311
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 32306 .summary) (.transfer 32310) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 32306 .summary)
      LeftBound32305.bound (LeftBound32305.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨49760⟩⟩) (rawTerms := some (Proof.Events126.exact32306RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound32305.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 32310)
      LeftBound32310.bound (LeftBound32310.actual selector witness) := by
  exact .transfer (LeftBound32310.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound32305.bound LeftBound32310.bound
def bound : CoeffClass := .finite ⟨32194504275408438756654574469120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32305.bound, LeftBound32310.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound32305.actual selector witness) * (LeftBound32310.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound32311

namespace LeftBound32322
def owner : Owner := ⟨.program ⟨257⟩, ⟨49078⟩⟩
def transferEvent : Nat := 32322
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 32320 .coefficient) (.value (.predecessor 1 32321 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 32320 .coefficient)
      LeftAuthority32318.bound (LeftAuthority32318.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events126.exact32319RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32318.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32318.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 32321 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority32318.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority32318.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority32318.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound32322

namespace LeftBound32326
def owner : Owner := ⟨.program ⟨257⟩, ⟨49079⟩⟩
def transferEvent : Nat := 32326
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 32324 .coefficient) (.predecessor 1 32325 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 32324 .coefficient)
      LeftBound32117.bound (LeftBound32117.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events125.exact32120RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32117.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32117.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 32325 .coefficient)
      LeftBound32322.bound (LeftBound32322.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events126.exact32323RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32322.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32322.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound32117.bound LeftBound32322.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32117.bound, LeftBound32322.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound32117.actual selector witness) * (LeftBound32322.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound32326

namespace LeftBound32327
def owner : Owner := ⟨.program ⟨257⟩, ⟨49079⟩⟩
def transferEvent : Nat := 32327
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨49076⟩⟩]⟩ [⟨.result 32319 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 32319 .coefficient)
      LeftAuthority32318.bound (LeftAuthority32318.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨49076⟩⟩) (rawTerms := some (Proof.Events126.exact32319RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32318.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32318.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority32318.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority32318.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority32318.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound32327

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
