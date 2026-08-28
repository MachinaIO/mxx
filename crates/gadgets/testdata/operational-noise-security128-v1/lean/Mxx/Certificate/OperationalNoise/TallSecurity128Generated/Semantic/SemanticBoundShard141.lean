import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard140

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound27306
def owner : Owner := ⟨.program ⟨257⟩, ⟨65995⟩⟩
def transferEvent : Nat := 27306
def frameStart : Nat := 26833
def rule : BoundRule := .sum [.predecessor 0 27304 .coefficient, .predecessor 1 27305 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27304 .coefficient)
      LeftBound27302.bound (LeftBound27302.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27303RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27302.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27302.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27305 .coefficient)
      LeftAuthority27036.bound (LeftAuthority27036.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events105.exact27037RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27036.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27036.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27302.bound, LeftAuthority27036.bound]
def bound : CoeffClass := .finite ⟨620, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27302.bound, LeftAuthority27036.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound27302.actual selector witness, LeftAuthority27036.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27306

namespace LeftBound27310
def owner : Owner := ⟨.program ⟨257⟩, ⟨65996⟩⟩
def transferEvent : Nat := 27310
def frameStart : Nat := 26833
def rule : BoundRule := .sum [.predecessor 0 27308 .coefficient, .predecessor 1 27309 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27308 .coefficient)
      LeftBound27306.bound (LeftBound27306.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27307RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27306.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27306.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27309 .coefficient)
      LeftAuthority27013.bound (LeftAuthority27013.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events105.exact27014RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27013.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27013.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27306.bound, LeftAuthority27013.bound]
def bound : CoeffClass := .finite ⟨682, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27306.bound, LeftAuthority27013.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound27306.actual selector witness, LeftAuthority27013.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27310

namespace LeftBound27314
def owner : Owner := ⟨.program ⟨257⟩, ⟨65997⟩⟩
def transferEvent : Nat := 27314
def frameStart : Nat := 26833
def rule : BoundRule := .sum [.predecessor 0 27312 .coefficient, .predecessor 1 27313 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27312 .coefficient)
      LeftBound27310.bound (LeftBound27310.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27311RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27310.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27310.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27313 .coefficient)
      LeftAuthority26990.bound (LeftAuthority26990.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events105.exact26991RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26990.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26990.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27310.bound, LeftAuthority26990.bound]
def bound : CoeffClass := .finite ⟨744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27310.bound, LeftAuthority26990.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound27310.actual selector witness, LeftAuthority26990.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27314

namespace LeftBound27318
def owner : Owner := ⟨.program ⟨257⟩, ⟨65998⟩⟩
def transferEvent : Nat := 27318
def frameStart : Nat := 26833
def rule : BoundRule := .sum [.predecessor 0 27316 .coefficient, .predecessor 1 27317 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27316 .coefficient)
      LeftBound27314.bound (LeftBound27314.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27315RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27314.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27314.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27317 .coefficient)
      LeftAuthority26967.bound (LeftAuthority26967.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events105.exact26968RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26967.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26967.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27314.bound, LeftAuthority26967.bound]
def bound : CoeffClass := .finite ⟨807, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27314.bound, LeftAuthority26967.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound27314.actual selector witness, LeftAuthority26967.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27318

namespace LeftBound27322
def owner : Owner := ⟨.program ⟨257⟩, ⟨65999⟩⟩
def transferEvent : Nat := 27322
def frameStart : Nat := 26833
def rule : BoundRule := .sum [.predecessor 0 27320 .coefficient, .predecessor 1 27321 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27320 .coefficient)
      LeftBound27318.bound (LeftBound27318.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27319RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27318.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27318.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27321 .coefficient)
      LeftAuthority26944.bound (LeftAuthority26944.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events105.exact26945RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26944.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26944.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27318.bound, LeftAuthority26944.bound]
def bound : CoeffClass := .finite ⟨870, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27318.bound, LeftAuthority26944.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound27318.actual selector witness, LeftAuthority26944.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27322

namespace LeftBound27326
def owner : Owner := ⟨.program ⟨257⟩, ⟨66000⟩⟩
def transferEvent : Nat := 27326
def frameStart : Nat := 26833
def rule : BoundRule := .sum [.predecessor 0 27324 .coefficient, .predecessor 1 27325 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27324 .coefficient)
      LeftBound27322.bound (LeftBound27322.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27323RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27322.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27322.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27325 .coefficient)
      LeftAuthority26921.bound (LeftAuthority26921.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events105.exact26922RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26921.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26921.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27322.bound, LeftAuthority26921.bound]
def bound : CoeffClass := .finite ⟨933, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27322.bound, LeftAuthority26921.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound27322.actual selector witness, LeftAuthority26921.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27326

namespace LeftBound27330
def owner : Owner := ⟨.program ⟨257⟩, ⟨66001⟩⟩
def transferEvent : Nat := 27330
def frameStart : Nat := 26833
def rule : BoundRule := .sum [.predecessor 0 27328 .coefficient, .predecessor 1 27329 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27328 .coefficient)
      LeftBound27326.bound (LeftBound27326.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27327RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27326.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27326.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27329 .coefficient)
      LeftAuthority26898.bound (LeftAuthority26898.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events105.exact26899RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26898.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26898.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27326.bound, LeftAuthority26898.bound]
def bound : CoeffClass := .finite ⟨996, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27326.bound, LeftAuthority26898.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound27326.actual selector witness, LeftAuthority26898.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27330

namespace LeftBound27334
def owner : Owner := ⟨.program ⟨257⟩, ⟨66002⟩⟩
def transferEvent : Nat := 27334
def frameStart : Nat := 26833
def rule : BoundRule := .sum [.predecessor 0 27332 .coefficient, .predecessor 1 27333 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27332 .coefficient)
      LeftBound27330.bound (LeftBound27330.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27331RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27330.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27330.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27333 .coefficient)
      LeftAuthority26875.bound (LeftAuthority26875.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events104.exact26876RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26875.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26875.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27330.bound, LeftAuthority26875.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27330.bound, LeftAuthority26875.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound27330.actual selector witness, LeftAuthority26875.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27334

namespace LeftBound27337
def owner : Owner := ⟨.program ⟨257⟩, ⟨66003⟩⟩
def transferEvent : Nat := 27337
def frameStart : Nat := 26833
def rule : BoundRule := .identity (.predecessor 0 27336 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27336 .coefficient)
      LeftBound27334.bound (LeftBound27334.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27335RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27334.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27334.derived selector witness)

def rawBound : CoeffClass := LeftBound27334.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27334.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound27334.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound27337

namespace LeftBound27354
def owner : Owner := ⟨.program ⟨257⟩, ⟨69051⟩⟩
def transferEvent : Nat := 27354
def frameStart : Nat := 26833
def rule : BoundRule := .sum [.predecessor 0 27352 .coefficient, .predecessor 1 27353 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27352 .coefficient)
      LeftBound27337.bound (LeftBound27337.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound27337.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27353 .coefficient)
      LeftAuthority27350.bound (LeftAuthority27350.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority27350.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27337.bound, LeftAuthority27350.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27337.bound, LeftAuthority27350.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound27337.actual selector witness, LeftAuthority27350.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27354

namespace LeftBound27357
def owner : Owner := ⟨.program ⟨257⟩, ⟨69052⟩⟩
def transferEvent : Nat := 27357
def frameStart : Nat := 26833
def rule : BoundRule := .identity (.predecessor 0 27356 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27356 .coefficient)
      LeftBound27354.bound (LeftBound27354.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound27354.derived selector witness)

def rawBound : CoeffClass := LeftBound27354.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27354.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound27354.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound27357

namespace LeftBound27363
def owner : Owner := ⟨.program ⟨257⟩, ⟨69053⟩⟩
def transferEvent : Nat := 27363
def frameStart : Nat := 26833
def rule : BoundRule := .product (.predecessor 0 27361 .coefficient) (.predecessor 1 27362 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27361 .coefficient)
      LeftAuthority27359.bound (LeftAuthority27359.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27360RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27359.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27359.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27362 .coefficient)
      LeftBound27357.bound (LeftBound27357.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27358RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27357.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27357.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority27359.bound LeftBound27357.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27359.bound, LeftBound27357.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority27359.actual selector witness) * (LeftBound27357.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27363

namespace LeftBound27439
def owner : Owner := ⟨.program ⟨257⟩, ⟨7309⟩⟩
def transferEvent : Nat := 27439
def frameStart : Nat := 26833
def rule : BoundRule := .sum [.predecessor 0 27437 .coefficient, .predecessor 1 27438 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27437 .coefficient)
      LeftAuthority27435.bound (LeftAuthority27435.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27436RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27435.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27435.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27438 .coefficient)
      LeftAuthority27432.bound (LeftAuthority27432.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27433RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27432.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27432.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority27435.bound, LeftAuthority27432.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27435.bound, LeftAuthority27432.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority27435.actual selector witness, LeftAuthority27432.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27439

namespace LeftBound27443
def owner : Owner := ⟨.program ⟨257⟩, ⟨7310⟩⟩
def transferEvent : Nat := 27443
def frameStart : Nat := 26833
def rule : BoundRule := .sum [.predecessor 0 27441 .coefficient, .predecessor 1 27442 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27441 .coefficient)
      LeftBound27439.bound (LeftBound27439.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27440RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27439.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27439.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27442 .coefficient)
      LeftAuthority27429.bound (LeftAuthority27429.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27430RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27429.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27429.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27439.bound, LeftAuthority27429.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27439.bound, LeftAuthority27429.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound27439.actual selector witness, LeftAuthority27429.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27443

namespace LeftBound27447
def owner : Owner := ⟨.program ⟨257⟩, ⟨7311⟩⟩
def transferEvent : Nat := 27447
def frameStart : Nat := 26833
def rule : BoundRule := .sum [.predecessor 0 27445 .coefficient, .predecessor 1 27446 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27445 .coefficient)
      LeftBound27443.bound (LeftBound27443.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27444RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27443.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27443.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27446 .coefficient)
      LeftAuthority27426.bound (LeftAuthority27426.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27427RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27426.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27426.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27443.bound, LeftAuthority27426.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27443.bound, LeftAuthority27426.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound27443.actual selector witness, LeftAuthority27426.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27447

namespace LeftBound27451
def owner : Owner := ⟨.program ⟨257⟩, ⟨7312⟩⟩
def transferEvent : Nat := 27451
def frameStart : Nat := 26833
def rule : BoundRule := .sum [.predecessor 0 27449 .coefficient, .predecessor 1 27450 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27449 .coefficient)
      LeftBound27447.bound (LeftBound27447.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27448RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27447.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27447.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27450 .coefficient)
      LeftAuthority27423.bound (LeftAuthority27423.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27424RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27423.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27423.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27447.bound, LeftAuthority27423.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27447.bound, LeftAuthority27423.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound27447.actual selector witness, LeftAuthority27423.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27451

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
