import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1508

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound224278
def owner : Owner := ⟨.program ⟨257⟩, ⟨38702⟩⟩
def transferEvent : Nat := 224278
def frameStart : Nat := 224228
def rule : BoundRule := .sum [.predecessor 0 224276 .coefficient, .predecessor 1 224277 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 224276 .coefficient)
      LeftBound224261.bound (LeftBound224261.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound224261.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 224277 .coefficient)
      LeftAuthority224274.bound (LeftAuthority224274.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority224274.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound224261.bound, LeftAuthority224274.bound]
def bound : CoeffClass := .finite ⟨1764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound224261.bound, LeftAuthority224274.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound224261.actual selector witness, LeftAuthority224274.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound224278

namespace LeftBound224281
def owner : Owner := ⟨.program ⟨257⟩, ⟨38703⟩⟩
def transferEvent : Nat := 224281
def frameStart : Nat := 224228
def rule : BoundRule := .identity (.predecessor 0 224280 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 224280 .coefficient)
      LeftBound224278.bound (LeftBound224278.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound224278.derived selector witness)

def rawBound : CoeffClass := LeftBound224278.bound
def bound : CoeffClass := .finite ⟨1764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound224278.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound224278.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound224281

namespace LeftBound224287
def owner : Owner := ⟨.program ⟨257⟩, ⟨38704⟩⟩
def transferEvent : Nat := 224287
def frameStart : Nat := 224228
def rule : BoundRule := .product (.predecessor 0 224285 .coefficient) (.predecessor 1 224286 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 224285 .coefficient)
      LeftAuthority224283.bound (LeftAuthority224283.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events876.exact224284RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority224283.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority224283.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 224286 .coefficient)
      LeftBound224281.bound (LeftBound224281.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events876.exact224282RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound224281.bound, RecordedBoundRefines] <;> decide)
      (LeftBound224281.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority224283.bound LeftBound224281.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority224283.bound, LeftBound224281.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority224283.actual selector witness) * (LeftBound224281.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound224287

namespace LeftBound224303
def owner : Owner := ⟨.program ⟨257⟩, ⟨9554⟩⟩
def transferEvent : Nat := 224303
def frameStart : Nat := 224228
def rule : BoundRule := .scale (.predecessor 0 224301 .coefficient) (.value (.predecessor 1 224302 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 224301 .coefficient)
      LeftAuthority224299.bound (LeftAuthority224299.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events876.exact224300RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority224299.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority224299.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 224302 .coefficient)
      LeftAuthority224290.bound (LeftAuthority224290.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority224290.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority224299.bound LeftAuthority224290.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority224299.bound, LeftAuthority224290.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority224299.actual selector witness) * (LeftAuthority224290.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound224303

namespace LeftBound224306
def owner : Owner := ⟨.program ⟨257⟩, ⟨7298⟩⟩
def transferEvent : Nat := 224306
def frameStart : Nat := 224228
def rule : BoundRule := .identity (.predecessor 0 224305 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 224305 .coefficient)
      LeftAuthority224293.bound (LeftAuthority224293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events876.exact224294RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority224293.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority224293.derived selector witness)

def rawBound : CoeffClass := LeftAuthority224293.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority224293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority224293.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound224306

namespace LeftBound224310
def owner : Owner := ⟨.program ⟨257⟩, ⟨9555⟩⟩
def transferEvent : Nat := 224310
def frameStart : Nat := 224228
def rule : BoundRule := .product (.predecessor 0 224308 .coefficient) (.predecessor 1 224309 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 224308 .coefficient)
      LeftBound224306.bound (LeftBound224306.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events876.exact224307RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound224306.bound, RecordedBoundRefines] <;> decide)
      (LeftBound224306.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 224309 .coefficient)
      LeftBound224303.bound (LeftBound224303.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events876.exact224304RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound224303.bound, RecordedBoundRefines] <;> decide)
      (LeftBound224303.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound224306.bound LeftBound224303.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound224306.bound, LeftBound224303.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound224306.actual selector witness) * (LeftBound224303.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound224310

namespace LeftBound224315
def owner : Owner := ⟨.program ⟨257⟩, ⟨38705⟩⟩
def transferEvent : Nat := 224315
def frameStart : Nat := 224228
def rule : BoundRule := .sum [.predecessor 0 224313 .coefficient, .predecessor 1 224314 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 224313 .coefficient)
      LeftBound224310.bound (LeftBound224310.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events876.exact224312RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound224310.bound, RecordedBoundRefines] <;> decide)
      (LeftBound224310.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 224314 .coefficient)
      LeftBound224287.bound (LeftBound224287.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events876.exact224289RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound224287.bound, RecordedBoundRefines] <;> decide)
      (LeftBound224287.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound224310.bound, LeftBound224287.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound224310.bound, LeftBound224287.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound224310.actual selector witness, LeftBound224287.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound224315

namespace LeftBound224319
def owner : Owner := ⟨.program ⟨257⟩, ⟨38931⟩⟩
def transferEvent : Nat := 224319
def frameStart : Nat := 224228
def rule : BoundRule := .product (.predecessor 0 224317 .coefficient) (.predecessor 1 224318 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 224317 .coefficient)
      LeftBound224315.bound (LeftBound224315.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events876.exact224316RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound224315.bound, RecordedBoundRefines] <;> decide)
      (LeftBound224315.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 224318 .coefficient)
      LeftAuthority224272.bound (LeftAuthority224272.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events876.exact224273RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority224272.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority224272.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound224315.bound LeftAuthority224272.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound224315.bound, LeftAuthority224272.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound224315.actual selector witness) * (LeftAuthority224272.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound224319

namespace LeftBound224330
def owner : Owner := ⟨.program ⟨257⟩, ⟨37422⟩⟩
def transferEvent : Nat := 224330
def frameStart : Nat := 224228
def rule : BoundRule := .product (.predecessor 0 224328 .coefficient) (.predecessor 1 224329 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 224328 .coefficient)
      LeftAuthority224283.bound (LeftAuthority224283.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events876.exact224284RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority224283.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority224283.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 224329 .coefficient)
      LeftAuthority224326.bound (LeftAuthority224326.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events876.exact224327RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority224326.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority224326.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority224283.bound LeftAuthority224326.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority224283.bound, LeftAuthority224326.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority224283.actual selector witness) * (LeftAuthority224326.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound224330

namespace LeftBound224338
def owner : Owner := ⟨.program ⟨257⟩, ⟨37423⟩⟩
def transferEvent : Nat := 224338
def frameStart : Nat := 224228
def rule : BoundRule := .sum [.predecessor 0 224336 .coefficient, .predecessor 1 224337 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 224336 .coefficient)
      LeftAuthority224334.bound (LeftAuthority224334.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events876.exact224335RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority224334.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority224334.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 224337 .coefficient)
      LeftBound224330.bound (LeftBound224330.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events876.exact224332RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound224330.bound, RecordedBoundRefines] <;> decide)
      (LeftBound224330.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority224334.bound, LeftBound224330.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority224334.bound, LeftBound224330.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority224334.actual selector witness, LeftBound224330.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound224338

namespace LeftBound224342
def owner : Owner := ⟨.program ⟨257⟩, ⟨38932⟩⟩
def transferEvent : Nat := 224342
def frameStart : Nat := 224228
def rule : BoundRule := .sum [.predecessor 0 224340 .coefficient, .predecessor 1 224341 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 224340 .coefficient)
      LeftBound224338.bound (LeftBound224338.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events876.exact224339RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound224338.bound, RecordedBoundRefines] <;> decide)
      (LeftBound224338.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 224341 .coefficient)
      LeftBound224319.bound (LeftBound224319.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events876.exact224324RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound224319.bound, RecordedBoundRefines] <;> decide)
      (LeftBound224319.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound224338.bound, LeftBound224319.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound224338.bound, LeftBound224319.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound224338.actual selector witness, LeftBound224319.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound224342

namespace LeftBound224355
def owner : Owner := ⟨.program ⟨257⟩, ⟨38930⟩⟩
def transferEvent : Nat := 224355
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 224353 .coefficient, .predecessor 1 224354 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 224353 .coefficient)
      LeftBound224176.bound (LeftBound224176.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events876.exact224352RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound224176.bound, RecordedBoundRefines] <;> decide)
      (LeftBound224176.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 224354 .coefficient)
      LeftBound224159.bound (LeftBound224159.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events875.exact224166RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound224159.bound, RecordedBoundRefines] <;> decide)
      (LeftBound224159.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound224176.bound, LeftBound224159.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound224176.bound, LeftBound224159.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound224176.actual selector witness, LeftBound224159.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound224355

namespace LeftBound224358
def owner : Owner := ⟨.program ⟨257⟩, ⟨38930⟩⟩
def transferEvent : Nat := 224358
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 224352 .summary, .result 224166 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 224352 .summary)
      LeftBound224178.bound (LeftBound224178.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨37862⟩⟩) (rawTerms := some (Proof.Events876.exact224352RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound224178.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 224166 .summary)
      LeftBound224161.bound (LeftBound224161.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨38929⟩⟩) (rawTerms := some (Proof.Events875.exact224166RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound224161.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound224178.bound, LeftBound224161.bound]
def bound : CoeffClass := .finite ⟨2998182198162866044928, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound224178.bound, LeftBound224161.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound224178.actual selector witness, LeftBound224161.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound224358

namespace LeftBound224362
def owner : Owner := ⟨.program ⟨257⟩, ⟨39286⟩⟩
def transferEvent : Nat := 224362
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 224360 .coefficient) (.predecessor 1 224361 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 224360 .coefficient)
      LeftBound224355.bound (LeftBound224355.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events876.exact224359RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound224355.bound, RecordedBoundRefines] <;> decide)
      (LeftBound224355.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 224361 .coefficient)
      LeftAuthority224081.bound (LeftAuthority224081.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events875.exact224082RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority224081.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority224081.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound224355.bound LeftAuthority224081.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound224355.bound, LeftAuthority224081.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound224355.actual selector witness) * (LeftAuthority224081.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound224362

namespace LeftBound224363
def owner : Owner := ⟨.program ⟨257⟩, ⟨39286⟩⟩
def transferEvent : Nat := 224363
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨39284⟩⟩]⟩ [⟨.result 224082 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 224082 .coefficient)
      LeftAuthority224081.bound (LeftAuthority224081.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨39284⟩⟩) (rawTerms := some (Proof.Events875.exact224082RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority224081.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority224081.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority224081.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority224081.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority224081.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound224363

namespace LeftBound224364
def owner : Owner := ⟨.program ⟨257⟩, ⟨39286⟩⟩
def transferEvent : Nat := 224364
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 224359 .summary) (.transfer 224363) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 224359 .summary)
      LeftBound224358.bound (LeftBound224358.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨38930⟩⟩) (rawTerms := some (Proof.Events876.exact224359RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound224358.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 224363)
      LeftBound224363.bound (LeftBound224363.actual selector witness) := by
  exact .transfer (LeftBound224363.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound224358.bound LeftBound224363.bound
def bound : CoeffClass := .finite ⟨32192736221397252361486566686720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound224358.bound, LeftBound224363.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound224358.actual selector witness) * (LeftBound224363.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound224364

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
