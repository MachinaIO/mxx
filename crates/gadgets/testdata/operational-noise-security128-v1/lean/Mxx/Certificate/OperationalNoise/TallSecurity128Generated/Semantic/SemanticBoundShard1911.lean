import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1910

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound282269
def owner : Owner := ⟨.program ⟨257⟩, ⟨39651⟩⟩
def transferEvent : Nat := 282269
def frameStart : Nat := 282240
def rule : BoundRule := .product (.predecessor 0 282267 .coefficient) (.predecessor 1 282268 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 282267 .coefficient)
      LeftAuthority282265.bound (LeftAuthority282265.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1102.exact282266RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority282265.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority282265.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 282268 .coefficient)
      LeftAuthority282262.bound (LeftAuthority282262.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1102.exact282263RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority282262.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority282262.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority282265.bound LeftAuthority282262.bound
def bound : CoeffClass := .finite ⟨2116, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority282265.bound, LeftAuthority282262.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority282265.actual selector witness) * (LeftAuthority282262.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound282269

namespace LeftBound282273
def owner : Owner := ⟨.program ⟨257⟩, ⟨39652⟩⟩
def transferEvent : Nat := 282273
def frameStart : Nat := 282240
def rule : BoundRule := .identity (.predecessor 0 282272 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 282272 .coefficient)
      LeftBound282269.bound (LeftBound282269.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1102.exact282271RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound282269.bound, RecordedBoundRefines] <;> decide)
      (LeftBound282269.derived selector witness)

def rawBound : CoeffClass := LeftBound282269.bound
def bound : CoeffClass := .finite ⟨2116, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound282269.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound282269.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound282273

namespace LeftBound282290
def owner : Owner := ⟨.program ⟨257⟩, ⟨41362⟩⟩
def transferEvent : Nat := 282290
def frameStart : Nat := 282240
def rule : BoundRule := .sum [.predecessor 0 282288 .coefficient, .predecessor 1 282289 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 282288 .coefficient)
      LeftBound282273.bound (LeftBound282273.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound282273.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 282289 .coefficient)
      LeftAuthority282286.bound (LeftAuthority282286.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority282286.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound282273.bound, LeftAuthority282286.bound]
def bound : CoeffClass := .finite ⟨2116, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound282273.bound, LeftAuthority282286.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound282273.actual selector witness, LeftAuthority282286.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound282290

namespace LeftBound282293
def owner : Owner := ⟨.program ⟨257⟩, ⟨41363⟩⟩
def transferEvent : Nat := 282293
def frameStart : Nat := 282240
def rule : BoundRule := .identity (.predecessor 0 282292 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 282292 .coefficient)
      LeftBound282290.bound (LeftBound282290.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound282290.derived selector witness)

def rawBound : CoeffClass := LeftBound282290.bound
def bound : CoeffClass := .finite ⟨2116, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound282290.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound282290.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound282293

namespace LeftBound282299
def owner : Owner := ⟨.program ⟨257⟩, ⟨41364⟩⟩
def transferEvent : Nat := 282299
def frameStart : Nat := 282240
def rule : BoundRule := .product (.predecessor 0 282297 .coefficient) (.predecessor 1 282298 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 282297 .coefficient)
      LeftAuthority282295.bound (LeftAuthority282295.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1102.exact282296RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority282295.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority282295.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 282298 .coefficient)
      LeftBound282293.bound (LeftBound282293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1102.exact282294RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound282293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound282293.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority282295.bound LeftBound282293.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority282295.bound, LeftBound282293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority282295.actual selector witness) * (LeftBound282293.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound282299

namespace LeftBound282313
def owner : Owner := ⟨.program ⟨257⟩, ⟨9557⟩⟩
def transferEvent : Nat := 282313
def frameStart : Nat := 282240
def rule : BoundRule := .scale (.predecessor 0 282311 .coefficient) (.value (.predecessor 1 282312 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 282311 .coefficient)
      LeftAuthority282309.bound (LeftAuthority282309.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1102.exact282310RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority282309.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority282309.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 282312 .coefficient)
      LeftAuthority282243.bound (LeftAuthority282243.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority282243.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority282309.bound LeftAuthority282243.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority282309.bound, LeftAuthority282243.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority282309.actual selector witness) * (LeftAuthority282243.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound282313

namespace LeftBound282316
def owner : Owner := ⟨.program ⟨257⟩, ⟨7299⟩⟩
def transferEvent : Nat := 282316
def frameStart : Nat := 282240
def rule : BoundRule := .identity (.predecessor 0 282315 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 282315 .coefficient)
      LeftAuthority282303.bound (LeftAuthority282303.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1102.exact282304RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority282303.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority282303.derived selector witness)

def rawBound : CoeffClass := LeftAuthority282303.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority282303.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority282303.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound282316

namespace LeftBound282320
def owner : Owner := ⟨.program ⟨257⟩, ⟨9558⟩⟩
def transferEvent : Nat := 282320
def frameStart : Nat := 282240
def rule : BoundRule := .product (.predecessor 0 282318 .coefficient) (.predecessor 1 282319 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 282318 .coefficient)
      LeftBound282316.bound (LeftBound282316.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1102.exact282317RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound282316.bound, RecordedBoundRefines] <;> decide)
      (LeftBound282316.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 282319 .coefficient)
      LeftBound282313.bound (LeftBound282313.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1102.exact282314RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound282313.bound, RecordedBoundRefines] <;> decide)
      (LeftBound282313.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound282316.bound LeftBound282313.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound282316.bound, LeftBound282313.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound282316.actual selector witness) * (LeftBound282313.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound282320

namespace LeftBound282325
def owner : Owner := ⟨.program ⟨257⟩, ⟨41365⟩⟩
def transferEvent : Nat := 282325
def frameStart : Nat := 282240
def rule : BoundRule := .sum [.predecessor 0 282323 .coefficient, .predecessor 1 282324 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 282323 .coefficient)
      LeftBound282320.bound (LeftBound282320.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1102.exact282322RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound282320.bound, RecordedBoundRefines] <;> decide)
      (LeftBound282320.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 282324 .coefficient)
      LeftBound282299.bound (LeftBound282299.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1102.exact282301RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound282299.bound, RecordedBoundRefines] <;> decide)
      (LeftBound282299.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound282320.bound, LeftBound282299.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound282320.bound, LeftBound282299.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound282320.actual selector witness, LeftBound282299.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound282325

namespace LeftBound282329
def owner : Owner := ⟨.program ⟨257⟩, ⟨41556⟩⟩
def transferEvent : Nat := 282329
def frameStart : Nat := 282240
def rule : BoundRule := .product (.predecessor 0 282327 .coefficient) (.predecessor 1 282328 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 282327 .coefficient)
      LeftBound282325.bound (LeftBound282325.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1102.exact282326RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound282325.bound, RecordedBoundRefines] <;> decide)
      (LeftBound282325.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 282328 .coefficient)
      LeftAuthority282284.bound (LeftAuthority282284.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1102.exact282285RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority282284.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority282284.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound282325.bound LeftAuthority282284.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound282325.bound, LeftAuthority282284.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound282325.actual selector witness) * (LeftAuthority282284.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound282329

namespace LeftBound282340
def owner : Owner := ⟨.program ⟨257⟩, ⟨40062⟩⟩
def transferEvent : Nat := 282340
def frameStart : Nat := 282240
def rule : BoundRule := .product (.predecessor 0 282338 .coefficient) (.predecessor 1 282339 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 282338 .coefficient)
      LeftAuthority282295.bound (LeftAuthority282295.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1102.exact282296RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority282295.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority282295.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 282339 .coefficient)
      LeftAuthority282336.bound (LeftAuthority282336.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1102.exact282337RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority282336.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority282336.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority282295.bound LeftAuthority282336.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority282295.bound, LeftAuthority282336.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority282295.actual selector witness) * (LeftAuthority282336.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound282340

namespace LeftBound282348
def owner : Owner := ⟨.program ⟨257⟩, ⟨40063⟩⟩
def transferEvent : Nat := 282348
def frameStart : Nat := 282240
def rule : BoundRule := .sum [.predecessor 0 282346 .coefficient, .predecessor 1 282347 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 282346 .coefficient)
      LeftAuthority282344.bound (LeftAuthority282344.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1102.exact282345RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority282344.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority282344.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 282347 .coefficient)
      LeftBound282340.bound (LeftBound282340.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1102.exact282342RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound282340.bound, RecordedBoundRefines] <;> decide)
      (LeftBound282340.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority282344.bound, LeftBound282340.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority282344.bound, LeftBound282340.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority282344.actual selector witness, LeftBound282340.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound282348

namespace LeftBound282352
def owner : Owner := ⟨.program ⟨257⟩, ⟨41557⟩⟩
def transferEvent : Nat := 282352
def frameStart : Nat := 282240
def rule : BoundRule := .sum [.predecessor 0 282350 .coefficient, .predecessor 1 282351 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 282350 .coefficient)
      LeftBound282348.bound (LeftBound282348.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1102.exact282349RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound282348.bound, RecordedBoundRefines] <;> decide)
      (LeftBound282348.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 282351 .coefficient)
      LeftBound282329.bound (LeftBound282329.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1102.exact282334RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound282329.bound, RecordedBoundRefines] <;> decide)
      (LeftBound282329.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound282348.bound, LeftBound282329.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound282348.bound, LeftBound282329.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound282348.actual selector witness, LeftBound282329.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound282352

namespace LeftBound282365
def owner : Owner := ⟨.program ⟨257⟩, ⟨41555⟩⟩
def transferEvent : Nat := 282365
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 282363 .coefficient, .predecessor 1 282364 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 282363 .coefficient)
      LeftBound282188.bound (LeftBound282188.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1102.exact282362RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound282188.bound, RecordedBoundRefines] <;> decide)
      (LeftBound282188.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 282364 .coefficient)
      LeftBound282171.bound (LeftBound282171.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1102.exact282178RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound282171.bound, RecordedBoundRefines] <;> decide)
      (LeftBound282171.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound282188.bound, LeftBound282171.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound282188.bound, LeftBound282171.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound282188.actual selector witness, LeftBound282171.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound282365

namespace LeftBound282368
def owner : Owner := ⟨.program ⟨257⟩, ⟨41555⟩⟩
def transferEvent : Nat := 282368
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 282362 .summary, .result 282178 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 282362 .summary)
      LeftBound282190.bound (LeftBound282190.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨40492⟩⟩) (rawTerms := some (Proof.Events1102.exact282362RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound282190.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 282178 .summary)
      LeftBound282173.bound (LeftBound282173.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨41554⟩⟩) (rawTerms := some (Proof.Events1102.exact282178RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound282173.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound282190.bound, LeftBound282173.bound]
def bound : CoeffClass := .finite ⟨2998218789909838430208, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound282190.bound, LeftBound282173.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound282190.actual selector witness, LeftBound282173.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound282368

namespace LeftBound282372
def owner : Owner := ⟨.program ⟨257⟩, ⟨41841⟩⟩
def transferEvent : Nat := 282372
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 282370 .coefficient) (.predecessor 1 282371 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 282370 .coefficient)
      LeftBound282365.bound (LeftBound282365.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1103.exact282369RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound282365.bound, RecordedBoundRefines] <;> decide)
      (LeftBound282365.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 282371 .coefficient)
      LeftAuthority282093.bound (LeftAuthority282093.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1101.exact282094RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority282093.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority282093.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound282365.bound LeftAuthority282093.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound282365.bound, LeftAuthority282093.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound282365.actual selector witness) * (LeftAuthority282093.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound282372

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
