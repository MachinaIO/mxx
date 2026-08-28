import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard106
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1388
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1391
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1426

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound212278
def owner : Owner := ⟨.program ⟨257⟩, ⟨64286⟩⟩
def transferEvent : Nat := 212278
def frameStart : Nat := 212222
def rule : BoundRule := .sum [.predecessor 0 212276 .coefficient, .predecessor 1 212277 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 212276 .coefficient)
      LeftBound212261.bound (LeftBound212261.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound212261.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 212277 .coefficient)
      LeftAuthority212274.bound (LeftAuthority212274.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority212274.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound212261.bound, LeftAuthority212274.bound]
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound212261.bound, LeftAuthority212274.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound212261.actual selector witness, LeftAuthority212274.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound212278

namespace LeftBound212281
def owner : Owner := ⟨.program ⟨257⟩, ⟨64287⟩⟩
def transferEvent : Nat := 212281
def frameStart : Nat := 212222
def rule : BoundRule := .identity (.predecessor 0 212280 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 212280 .coefficient)
      LeftBound212278.bound (LeftBound212278.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound212278.derived selector witness)

def rawBound : CoeffClass := LeftBound212278.bound
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound212278.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound212278.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound212281

namespace LeftBound212287
def owner : Owner := ⟨.program ⟨257⟩, ⟨64288⟩⟩
def transferEvent : Nat := 212287
def frameStart : Nat := 212222
def rule : BoundRule := .product (.predecessor 0 212285 .coefficient) (.predecessor 1 212286 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 212285 .coefficient)
      LeftAuthority212283.bound (LeftAuthority212283.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events829.exact212284RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority212283.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority212283.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 212286 .coefficient)
      LeftBound212281.bound (LeftBound212281.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events829.exact212282RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound212281.bound, RecordedBoundRefines] <;> decide)
      (LeftBound212281.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority212283.bound LeftBound212281.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority212283.bound, LeftBound212281.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority212283.actual selector witness) * (LeftBound212281.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound212287

namespace LeftBound212295
def owner : Owner := ⟨.program ⟨257⟩, ⟨64289⟩⟩
def transferEvent : Nat := 212295
def frameStart : Nat := 212222
def rule : BoundRule := .sum [.predecessor 0 212293 .coefficient, .predecessor 1 212294 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 212293 .coefficient)
      LeftAuthority212291.bound (LeftAuthority212291.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events829.exact212292RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority212291.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority212291.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 212294 .coefficient)
      LeftBound212287.bound (LeftBound212287.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events829.exact212289RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound212287.bound, RecordedBoundRefines] <;> decide)
      (LeftBound212287.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority212291.bound, LeftBound212287.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority212291.bound, LeftBound212287.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority212291.actual selector witness, LeftBound212287.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound212295

namespace LeftBound212299
def owner : Owner := ⟨.program ⟨257⟩, ⟨64873⟩⟩
def transferEvent : Nat := 212299
def frameStart : Nat := 212222
def rule : BoundRule := .product (.predecessor 0 212297 .coefficient) (.predecessor 1 212298 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 212297 .coefficient)
      LeftBound212295.bound (LeftBound212295.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events829.exact212296RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound212295.bound, RecordedBoundRefines] <;> decide)
      (LeftBound212295.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 212298 .coefficient)
      LeftAuthority212272.bound (LeftAuthority212272.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events829.exact212273RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority212272.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority212272.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound212295.bound LeftAuthority212272.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound212295.bound, LeftAuthority212272.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound212295.actual selector witness) * (LeftAuthority212272.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound212299

namespace LeftBound212310
def owner : Owner := ⟨.program ⟨257⟩, ⟨63083⟩⟩
def transferEvent : Nat := 212310
def frameStart : Nat := 212222
def rule : BoundRule := .product (.predecessor 0 212308 .coefficient) (.predecessor 1 212309 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 212308 .coefficient)
      LeftAuthority212283.bound (LeftAuthority212283.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events829.exact212284RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority212283.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority212283.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 212309 .coefficient)
      LeftAuthority212306.bound (LeftAuthority212306.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events829.exact212307RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority212306.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority212306.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority212283.bound LeftAuthority212306.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority212283.bound, LeftAuthority212306.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority212283.actual selector witness) * (LeftAuthority212306.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound212310

namespace LeftBound212318
def owner : Owner := ⟨.program ⟨257⟩, ⟨63084⟩⟩
def transferEvent : Nat := 212318
def frameStart : Nat := 212222
def rule : BoundRule := .sum [.predecessor 0 212316 .coefficient, .predecessor 1 212317 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 212316 .coefficient)
      LeftAuthority212314.bound (LeftAuthority212314.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events829.exact212315RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority212314.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority212314.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 212317 .coefficient)
      LeftBound212310.bound (LeftBound212310.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events829.exact212312RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound212310.bound, RecordedBoundRefines] <;> decide)
      (LeftBound212310.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority212314.bound, LeftBound212310.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority212314.bound, LeftBound212310.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority212314.actual selector witness, LeftBound212310.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound212318

namespace LeftBound212322
def owner : Owner := ⟨.program ⟨257⟩, ⟨64877⟩⟩
def transferEvent : Nat := 212322
def frameStart : Nat := 212222
def rule : BoundRule := .sum [.predecessor 0 212320 .coefficient, .predecessor 1 212321 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 212320 .coefficient)
      LeftBound212318.bound (LeftBound212318.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events829.exact212319RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound212318.bound, RecordedBoundRefines] <;> decide)
      (LeftBound212318.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 212321 .coefficient)
      LeftBound212299.bound (LeftBound212299.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events829.exact212304RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound212299.bound, RecordedBoundRefines] <;> decide)
      (LeftBound212299.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound212318.bound, LeftBound212299.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound212318.bound, LeftBound212299.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound212318.actual selector witness, LeftBound212299.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound212322

namespace LeftBound212335
def owner : Owner := ⟨.program ⟨257⟩, ⟨64875⟩⟩
def transferEvent : Nat := 212335
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 212333 .coefficient, .predecessor 1 212334 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 212333 .coefficient)
      LeftBound212164.bound (LeftBound212164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events829.exact212332RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound212164.bound, RecordedBoundRefines] <;> decide)
      (LeftBound212164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 212334 .coefficient)
      LeftBound212147.bound (LeftBound212147.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events828.exact212154RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound212147.bound, RecordedBoundRefines] <;> decide)
      (LeftBound212147.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound212164.bound, LeftBound212147.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound212164.bound, LeftBound212147.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound212164.actual selector witness, LeftBound212147.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound212335

namespace LeftBound212338
def owner : Owner := ⟨.program ⟨257⟩, ⟨64875⟩⟩
def transferEvent : Nat := 212338
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 212332 .summary, .result 212154 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 212332 .summary)
      LeftBound212166.bound (LeftBound212166.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨63679⟩⟩) (rawTerms := some (Proof.Events829.exact212332RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound212166.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 212154 .summary)
      LeftBound212149.bound (LeftBound212149.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64874⟩⟩) (rawTerms := some (Proof.Events828.exact212154RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound212149.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound212166.bound, LeftBound212149.bound]
def bound : CoeffClass := .finite ⟨32190771716940580661919523012608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound212166.bound, LeftBound212149.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound212166.actual selector witness, LeftBound212149.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound212338

namespace LeftBound212362
def owner : Owner := ⟨.program ⟨257⟩, ⟨25251⟩⟩
def transferEvent : Nat := 212362
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 212360 .coefficient) (.predecessor 1 212361 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 212360 .coefficient)
      LeftAuthority10047.bound (LeftAuthority10047.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10048RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10047.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10047.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 212361 .coefficient)
      LeftBound207526.bound (LeftBound207526.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events810.exact207528RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207526.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207526.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority10047.bound LeftBound207526.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10047.bound, LeftBound207526.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority10047.actual selector witness) * (LeftBound207526.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound212362

namespace LeftBound212367
def owner : Owner := ⟨.program ⟨257⟩, ⟨8580⟩⟩
def transferEvent : Nat := 212367
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 212365 .coefficient) (.predecessor 1 212366 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 212365 .coefficient)
      LeftBound207397.bound (LeftBound207397.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events810.exact207398RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207397.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207397.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 212366 .coefficient)
      LeftBound22089.bound (LeftBound22089.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events086.exact22090RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22089.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22089.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound207397.bound LeftBound22089.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207397.bound, LeftBound22089.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound207397.actual selector witness) * (LeftBound22089.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound212367

namespace LeftBound212372
def owner : Owner := ⟨.program ⟨257⟩, ⟨25252⟩⟩
def transferEvent : Nat := 212372
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 212370 .coefficient, .predecessor 1 212371 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 212370 .coefficient)
      LeftBound212367.bound (LeftBound212367.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events829.exact212369RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound212367.bound, RecordedBoundRefines] <;> decide)
      (LeftBound212367.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 212371 .coefficient)
      LeftBound212362.bound (LeftBound212362.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events829.exact212364RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound212362.bound, RecordedBoundRefines] <;> decide)
      (LeftBound212362.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound212367.bound, LeftBound212362.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound212367.bound, LeftBound212362.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound212367.actual selector witness, LeftBound212362.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound212372

namespace LeftBound212376
def owner : Owner := ⟨.program ⟨257⟩, ⟨25253⟩⟩
def transferEvent : Nat := 212376
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 212374 .coefficient, .predecessor 1 212375 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 212374 .coefficient)
      LeftBound212372.bound (LeftBound212372.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events829.exact212373RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound212372.bound, RecordedBoundRefines] <;> decide)
      (LeftBound212372.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 212375 .coefficient)
      LeftBound22081.bound (LeftBound22081.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events086.exact22082RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22081.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22081.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound212372.bound, LeftBound22081.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound212372.bound, LeftBound22081.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound212372.actual selector witness, LeftBound22081.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound212376

namespace LeftBound212377
def owner : Owner := ⟨.program ⟨257⟩, ⟨25253⟩⟩
def transferEvent : Nat := 212377
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨100⟩⟩]⟩ [⟨.result 22082 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 22082 .coefficient)
      LeftBound22081.bound (LeftBound22081.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨100⟩⟩) (rawTerms := some (Proof.Events086.exact22082RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22081.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22081.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound22081.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22081.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound22081.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound212377

namespace LeftBound212382
def owner : Owner := ⟨.program ⟨257⟩, ⟨59488⟩⟩
def transferEvent : Nat := 212382
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 212380 .coefficient) (.predecessor 1 212381 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 212380 .coefficient)
      LeftBound212376.bound (LeftBound212376.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events829.exact212379RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound212376.bound, RecordedBoundRefines] <;> decide)
      (LeftBound212376.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 212381 .coefficient)
      LeftAuthority10050.bound (LeftAuthority10050.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10051RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10050.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10050.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound212376.bound LeftAuthority10050.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound212376.bound, LeftAuthority10050.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound212376.actual selector witness) * (LeftAuthority10050.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound212382

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
