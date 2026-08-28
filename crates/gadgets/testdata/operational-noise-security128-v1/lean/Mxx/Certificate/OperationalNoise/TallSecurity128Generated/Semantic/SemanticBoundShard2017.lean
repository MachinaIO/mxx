import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1998
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2016

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound297455
def owner : Owner := ⟨.program ⟨257⟩, ⟨35988⟩⟩
def transferEvent : Nat := 297455
def frameStart : Nat := 297408
def rule : BoundRule := .product (.predecessor 0 297453 .coefficient) (.predecessor 1 297454 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 297453 .coefficient)
      LeftAuthority297451.bound (LeftAuthority297451.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1161.exact297452RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority297451.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority297451.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 297454 .coefficient)
      LeftBound297449.bound (LeftBound297449.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1161.exact297450RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound297449.bound, RecordedBoundRefines] <;> decide)
      (LeftBound297449.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority297451.bound LeftBound297449.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority297451.bound, LeftBound297449.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority297451.actual selector witness) * (LeftBound297449.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound297455

namespace LeftBound297471
def owner : Owner := ⟨.program ⟨257⟩, ⟨9551⟩⟩
def transferEvent : Nat := 297471
def frameStart : Nat := 297408
def rule : BoundRule := .scale (.predecessor 0 297469 .coefficient) (.value (.predecessor 1 297470 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 297469 .coefficient)
      LeftAuthority297467.bound (LeftAuthority297467.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1161.exact297468RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority297467.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority297467.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 297470 .coefficient)
      LeftAuthority297458.bound (LeftAuthority297458.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority297458.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority297467.bound LeftAuthority297458.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority297467.bound, LeftAuthority297458.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority297467.actual selector witness) * (LeftAuthority297458.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound297471

namespace LeftBound297474
def owner : Owner := ⟨.program ⟨257⟩, ⟨7297⟩⟩
def transferEvent : Nat := 297474
def frameStart : Nat := 297408
def rule : BoundRule := .identity (.predecessor 0 297473 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 297473 .coefficient)
      LeftAuthority297461.bound (LeftAuthority297461.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1161.exact297462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority297461.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority297461.derived selector witness)

def rawBound : CoeffClass := LeftAuthority297461.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority297461.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority297461.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound297474

namespace LeftBound297478
def owner : Owner := ⟨.program ⟨257⟩, ⟨9552⟩⟩
def transferEvent : Nat := 297478
def frameStart : Nat := 297408
def rule : BoundRule := .product (.predecessor 0 297476 .coefficient) (.predecessor 1 297477 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 297476 .coefficient)
      LeftBound297474.bound (LeftBound297474.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1162.exact297475RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound297474.bound, RecordedBoundRefines] <;> decide)
      (LeftBound297474.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 297477 .coefficient)
      LeftBound297471.bound (LeftBound297471.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1162.exact297472RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound297471.bound, RecordedBoundRefines] <;> decide)
      (LeftBound297471.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound297474.bound LeftBound297471.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound297474.bound, LeftBound297471.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound297474.actual selector witness) * (LeftBound297471.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound297478

namespace LeftBound297483
def owner : Owner := ⟨.program ⟨257⟩, ⟨35989⟩⟩
def transferEvent : Nat := 297483
def frameStart : Nat := 297408
def rule : BoundRule := .sum [.predecessor 0 297481 .coefficient, .predecessor 1 297482 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 297481 .coefficient)
      LeftBound297478.bound (LeftBound297478.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1162.exact297480RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound297478.bound, RecordedBoundRefines] <;> decide)
      (LeftBound297478.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 297482 .coefficient)
      LeftBound297455.bound (LeftBound297455.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1161.exact297457RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound297455.bound, RecordedBoundRefines] <;> decide)
      (LeftBound297455.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound297478.bound, LeftBound297455.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound297478.bound, LeftBound297455.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound297478.actual selector witness, LeftBound297455.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound297483

namespace LeftBound297487
def owner : Owner := ⟨.program ⟨257⟩, ⟨36152⟩⟩
def transferEvent : Nat := 297487
def frameStart : Nat := 297408
def rule : BoundRule := .product (.predecessor 0 297485 .coefficient) (.predecessor 1 297486 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 297485 .coefficient)
      LeftBound297483.bound (LeftBound297483.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1162.exact297484RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound297483.bound, RecordedBoundRefines] <;> decide)
      (LeftBound297483.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 297486 .coefficient)
      LeftAuthority297440.bound (LeftAuthority297440.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1161.exact297441RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority297440.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority297440.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound297483.bound LeftAuthority297440.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound297483.bound, LeftAuthority297440.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound297483.actual selector witness) * (LeftAuthority297440.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound297487

namespace LeftBound297498
def owner : Owner := ⟨.program ⟨257⟩, ⟨34670⟩⟩
def transferEvent : Nat := 297498
def frameStart : Nat := 297408
def rule : BoundRule := .product (.predecessor 0 297496 .coefficient) (.predecessor 1 297497 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 297496 .coefficient)
      LeftAuthority297451.bound (LeftAuthority297451.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1161.exact297452RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority297451.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority297451.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 297497 .coefficient)
      LeftAuthority297494.bound (LeftAuthority297494.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1162.exact297495RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority297494.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority297494.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority297451.bound LeftAuthority297494.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority297451.bound, LeftAuthority297494.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority297451.actual selector witness) * (LeftAuthority297494.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound297498

namespace LeftBound297506
def owner : Owner := ⟨.program ⟨257⟩, ⟨34671⟩⟩
def transferEvent : Nat := 297506
def frameStart : Nat := 297408
def rule : BoundRule := .sum [.predecessor 0 297504 .coefficient, .predecessor 1 297505 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 297504 .coefficient)
      LeftAuthority297502.bound (LeftAuthority297502.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1162.exact297503RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority297502.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority297502.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 297505 .coefficient)
      LeftBound297498.bound (LeftBound297498.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1162.exact297500RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound297498.bound, RecordedBoundRefines] <;> decide)
      (LeftBound297498.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority297502.bound, LeftBound297498.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority297502.bound, LeftBound297498.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority297502.actual selector witness, LeftBound297498.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound297506

namespace LeftBound297510
def owner : Owner := ⟨.program ⟨257⟩, ⟨36153⟩⟩
def transferEvent : Nat := 297510
def frameStart : Nat := 297408
def rule : BoundRule := .sum [.predecessor 0 297508 .coefficient, .predecessor 1 297509 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 297508 .coefficient)
      LeftBound297506.bound (LeftBound297506.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1162.exact297507RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound297506.bound, RecordedBoundRefines] <;> decide)
      (LeftBound297506.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 297509 .coefficient)
      LeftBound297487.bound (LeftBound297487.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1162.exact297492RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound297487.bound, RecordedBoundRefines] <;> decide)
      (LeftBound297487.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound297506.bound, LeftBound297487.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound297506.bound, LeftBound297487.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound297506.actual selector witness, LeftBound297487.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound297510

namespace LeftBound297523
def owner : Owner := ⟨.program ⟨257⟩, ⟨36151⟩⟩
def transferEvent : Nat := 297523
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 297521 .coefficient, .predecessor 1 297522 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 297521 .coefficient)
      LeftBound297368.bound (LeftBound297368.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1162.exact297520RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound297368.bound, RecordedBoundRefines] <;> decide)
      (LeftBound297368.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 297522 .coefficient)
      LeftBound297351.bound (LeftBound297351.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1161.exact297358RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound297351.bound, RecordedBoundRefines] <;> decide)
      (LeftBound297351.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound297368.bound, LeftBound297351.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound297368.bound, LeftBound297351.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound297368.actual selector witness, LeftBound297351.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound297523

namespace LeftBound297526
def owner : Owner := ⟨.program ⟨257⟩, ⟨36151⟩⟩
def transferEvent : Nat := 297526
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 297520 .summary, .result 297358 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 297520 .summary)
      LeftBound297370.bound (LeftBound297370.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨35092⟩⟩) (rawTerms := some (Proof.Events1162.exact297520RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound297370.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 297358 .summary)
      LeftBound297353.bound (LeftBound297353.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36150⟩⟩) (rawTerms := some (Proof.Events1161.exact297358RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound297353.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound297370.bound, LeftBound297353.bound]
def bound : CoeffClass := .finite ⟨2998163902289379852288, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound297370.bound, LeftBound297353.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound297370.actual selector witness, LeftBound297353.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound297526

namespace LeftBound297530
def owner : Owner := ⟨.program ⟨257⟩, ⟨36381⟩⟩
def transferEvent : Nat := 297530
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 297528 .coefficient) (.predecessor 1 297529 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 297528 .coefficient)
      LeftBound297523.bound (LeftBound297523.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1162.exact297527RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound297523.bound, RecordedBoundRefines] <;> decide)
      (LeftBound297523.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 297529 .coefficient)
      LeftAuthority297273.bound (LeftAuthority297273.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1161.exact297274RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority297273.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority297273.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound297523.bound LeftAuthority297273.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound297523.bound, LeftAuthority297273.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound297523.actual selector witness) * (LeftAuthority297273.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound297530

namespace LeftBound297531
def owner : Owner := ⟨.program ⟨257⟩, ⟨36381⟩⟩
def transferEvent : Nat := 297531
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨36379⟩⟩]⟩ [⟨.result 297274 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 297274 .coefficient)
      LeftAuthority297273.bound (LeftAuthority297273.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨36379⟩⟩) (rawTerms := some (Proof.Events1161.exact297274RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority297273.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority297273.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority297273.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority297273.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority297273.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound297531

namespace LeftBound297532
def owner : Owner := ⟨.program ⟨257⟩, ⟨36381⟩⟩
def transferEvent : Nat := 297532
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 297527 .summary) (.transfer 297531) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 297527 .summary)
      LeftBound297526.bound (LeftBound297526.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36151⟩⟩) (rawTerms := some (Proof.Events1162.exact297527RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound297526.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 297531)
      LeftBound297531.bound (LeftBound297531.actual selector witness) := by
  exact .transfer (LeftBound297531.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound297526.bound LeftBound297531.bound
def bound : CoeffClass := .finite ⟨32192539770951564984245676933120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound297526.bound, LeftBound297531.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound297526.actual selector witness) * (LeftBound297531.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound297532

namespace LeftBound297543
def owner : Owner := ⟨.program ⟨257⟩, ⟨35298⟩⟩
def transferEvent : Nat := 297543
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 297541 .coefficient) (.value (.predecessor 1 297542 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 297541 .coefficient)
      LeftAuthority297539.bound (LeftAuthority297539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1162.exact297540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority297539.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority297539.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 297542 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority297539.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority297539.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority297539.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound297543

namespace LeftBound297547
def owner : Owner := ⟨.program ⟨257⟩, ⟨35299⟩⟩
def transferEvent : Nat := 297547
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 297545 .coefficient) (.predecessor 1 297546 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 297545 .coefficient)
      LeftBound295192.bound (LeftBound295192.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1153.exact295195RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound295192.bound, RecordedBoundRefines] <;> decide)
      (LeftBound295192.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 297546 .coefficient)
      LeftBound297543.bound (LeftBound297543.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1162.exact297544RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound297543.bound, RecordedBoundRefines] <;> decide)
      (LeftBound297543.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound295192.bound LeftBound297543.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound295192.bound, LeftBound297543.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound295192.actual selector witness) * (LeftBound297543.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound297547

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
