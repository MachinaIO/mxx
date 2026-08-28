import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard102
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1489
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1492
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1524

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound226438
def owner : Owner := ⟨.program ⟨257⟩, ⟨69006⟩⟩
def transferEvent : Nat := 226438
def frameStart : Nat := 226365
def rule : BoundRule := .sum [.predecessor 0 226436 .coefficient, .predecessor 1 226437 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 226436 .coefficient)
      LeftAuthority226434.bound (LeftAuthority226434.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events884.exact226435RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority226434.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority226434.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 226437 .coefficient)
      LeftBound226430.bound (LeftBound226430.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events884.exact226432RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound226430.bound, RecordedBoundRefines] <;> decide)
      (LeftBound226430.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority226434.bound, LeftBound226430.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority226434.bound, LeftBound226430.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority226434.actual selector witness, LeftBound226430.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound226438

namespace LeftBound226442
def owner : Owner := ⟨.program ⟨257⟩, ⟨70099⟩⟩
def transferEvent : Nat := 226442
def frameStart : Nat := 226365
def rule : BoundRule := .product (.predecessor 0 226440 .coefficient) (.predecessor 1 226441 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 226440 .coefficient)
      LeftBound226438.bound (LeftBound226438.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events884.exact226439RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound226438.bound, RecordedBoundRefines] <;> decide)
      (LeftBound226438.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 226441 .coefficient)
      LeftAuthority226415.bound (LeftAuthority226415.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events884.exact226416RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority226415.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority226415.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound226438.bound LeftAuthority226415.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound226438.bound, LeftAuthority226415.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound226438.actual selector witness) * (LeftAuthority226415.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound226442

namespace LeftBound226453
def owner : Owner := ⟨.program ⟨257⟩, ⟨66542⟩⟩
def transferEvent : Nat := 226453
def frameStart : Nat := 226365
def rule : BoundRule := .product (.predecessor 0 226451 .coefficient) (.predecessor 1 226452 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 226451 .coefficient)
      LeftAuthority226426.bound (LeftAuthority226426.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events884.exact226427RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority226426.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority226426.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 226452 .coefficient)
      LeftAuthority226449.bound (LeftAuthority226449.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events884.exact226450RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority226449.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority226449.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority226426.bound LeftAuthority226449.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority226426.bound, LeftAuthority226449.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority226426.actual selector witness) * (LeftAuthority226449.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound226453

namespace LeftBound226461
def owner : Owner := ⟨.program ⟨257⟩, ⟨66543⟩⟩
def transferEvent : Nat := 226461
def frameStart : Nat := 226365
def rule : BoundRule := .sum [.predecessor 0 226459 .coefficient, .predecessor 1 226460 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 226459 .coefficient)
      LeftAuthority226457.bound (LeftAuthority226457.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events884.exact226458RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority226457.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority226457.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 226460 .coefficient)
      LeftBound226453.bound (LeftBound226453.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events884.exact226455RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound226453.bound, RecordedBoundRefines] <;> decide)
      (LeftBound226453.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority226457.bound, LeftBound226453.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority226457.bound, LeftBound226453.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority226457.actual selector witness, LeftBound226453.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound226461

namespace LeftBound226465
def owner : Owner := ⟨.program ⟨257⟩, ⟨70111⟩⟩
def transferEvent : Nat := 226465
def frameStart : Nat := 226365
def rule : BoundRule := .sum [.predecessor 0 226463 .coefficient, .predecessor 1 226464 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 226463 .coefficient)
      LeftBound226461.bound (LeftBound226461.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events884.exact226462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound226461.bound, RecordedBoundRefines] <;> decide)
      (LeftBound226461.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 226464 .coefficient)
      LeftBound226442.bound (LeftBound226442.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events884.exact226447RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound226442.bound, RecordedBoundRefines] <;> decide)
      (LeftBound226442.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound226461.bound, LeftBound226442.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound226461.bound, LeftBound226442.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound226461.actual selector witness, LeftBound226442.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound226465

namespace LeftBound226478
def owner : Owner := ⟨.program ⟨257⟩, ⟨70101⟩⟩
def transferEvent : Nat := 226478
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 226476 .coefficient, .predecessor 1 226477 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 226476 .coefficient)
      LeftBound226307.bound (LeftBound226307.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events884.exact226475RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound226307.bound, RecordedBoundRefines] <;> decide)
      (LeftBound226307.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 226477 .coefficient)
      LeftBound226290.bound (LeftBound226290.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events883.exact226297RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound226290.bound, RecordedBoundRefines] <;> decide)
      (LeftBound226290.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound226307.bound, LeftBound226290.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound226307.bound, LeftBound226290.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound226307.actual selector witness, LeftBound226290.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound226478

namespace LeftBound226481
def owner : Owner := ⟨.program ⟨257⟩, ⟨70101⟩⟩
def transferEvent : Nat := 226481
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 226475 .summary, .result 226297 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 226475 .summary)
      LeftBound226309.bound (LeftBound226309.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨68060⟩⟩) (rawTerms := some (Proof.Events884.exact226475RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound226309.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 226297 .summary)
      LeftBound226292.bound (LeftBound226292.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70100⟩⟩) (rawTerms := some (Proof.Events883.exact226297RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound226292.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound226309.bound, LeftBound226292.bound]
def bound : CoeffClass := .finite ⟨32191361068277642793642192273408, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound226309.bound, LeftBound226292.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound226309.actual selector witness, LeftBound226292.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound226481

namespace LeftBound226505
def owner : Owner := ⟨.program ⟨257⟩, ⟨25479⟩⟩
def transferEvent : Nat := 226505
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 226503 .coefficient) (.predecessor 1 226504 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 226503 .coefficient)
      LeftAuthority10772.bound (LeftAuthority10772.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events042.exact10773RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10772.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10772.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 226504 .coefficient)
      LeftBound222151.bound (LeftBound222151.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events867.exact222153RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound222151.bound, RecordedBoundRefines] <;> decide)
      (LeftBound222151.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority10772.bound LeftBound222151.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10772.bound, LeftBound222151.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority10772.actual selector witness) * (LeftBound222151.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound226505

namespace LeftBound226510
def owner : Owner := ⟨.program ⟨257⟩, ⟨8467⟩⟩
def transferEvent : Nat := 226510
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 226508 .coefficient) (.predecessor 1 226509 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 226508 .coefficient)
      LeftBound222022.bound (LeftBound222022.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events867.exact222023RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound222022.bound, RecordedBoundRefines] <;> decide)
      (LeftBound222022.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 226509 .coefficient)
      LeftBound21588.bound (LeftBound21588.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21589RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21588.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21588.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound222022.bound LeftBound21588.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound222022.bound, LeftBound21588.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound222022.actual selector witness) * (LeftBound21588.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound226510

namespace LeftBound226515
def owner : Owner := ⟨.program ⟨257⟩, ⟨25480⟩⟩
def transferEvent : Nat := 226515
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 226513 .coefficient, .predecessor 1 226514 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 226513 .coefficient)
      LeftBound226510.bound (LeftBound226510.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events884.exact226512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound226510.bound, RecordedBoundRefines] <;> decide)
      (LeftBound226510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 226514 .coefficient)
      LeftBound226505.bound (LeftBound226505.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events884.exact226507RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound226505.bound, RecordedBoundRefines] <;> decide)
      (LeftBound226505.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound226510.bound, LeftBound226505.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound226510.bound, LeftBound226505.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound226510.actual selector witness, LeftBound226505.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound226515

namespace LeftBound226519
def owner : Owner := ⟨.program ⟨257⟩, ⟨25481⟩⟩
def transferEvent : Nat := 226519
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 226517 .coefficient, .predecessor 1 226518 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 226517 .coefficient)
      LeftBound226515.bound (LeftBound226515.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events884.exact226516RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound226515.bound, RecordedBoundRefines] <;> decide)
      (LeftBound226515.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 226518 .coefficient)
      LeftBound21580.bound (LeftBound21580.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21581RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21580.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21580.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound226515.bound, LeftBound21580.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound226515.bound, LeftBound21580.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound226515.actual selector witness, LeftBound21580.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound226519

namespace LeftBound226520
def owner : Owner := ⟨.program ⟨257⟩, ⟨25481⟩⟩
def transferEvent : Nat := 226520
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨101⟩⟩]⟩ [⟨.result 21581 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 21581 .coefficient)
      LeftBound21580.bound (LeftBound21580.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨101⟩⟩) (rawTerms := some (Proof.Events084.exact21581RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21580.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21580.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound21580.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21580.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound21580.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound226520

namespace LeftBound226525
def owner : Owner := ⟨.program ⟨257⟩, ⟨62441⟩⟩
def transferEvent : Nat := 226525
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 226523 .coefficient) (.predecessor 1 226524 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 226523 .coefficient)
      LeftBound226519.bound (LeftBound226519.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events884.exact226522RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound226519.bound, RecordedBoundRefines] <;> decide)
      (LeftBound226519.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 226524 .coefficient)
      LeftAuthority10775.bound (LeftAuthority10775.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events042.exact10776RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10775.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10775.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound226519.bound LeftAuthority10775.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound226519.bound, LeftAuthority10775.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound226519.actual selector witness) * (LeftAuthority10775.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound226525

namespace LeftBound226526
def owner : Owner := ⟨.program ⟨257⟩, ⟨62441⟩⟩
def transferEvent : Nat := 226526
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨62438⟩⟩], []⟩ [⟨.result 10776 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 10776 .coefficient)
      LeftAuthority10775.bound (LeftAuthority10775.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨62438⟩⟩) (rawTerms := some (Proof.Events042.exact10776RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10775.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10775.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority10775.bound []
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10775.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority10775.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound226526

namespace LeftBound226527
def owner : Owner := ⟨.program ⟨257⟩, ⟨62441⟩⟩
def transferEvent : Nat := 226527
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 226522 .summary) (.transfer 226526) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 226522 .summary)
      LeftBound226520.bound (LeftBound226520.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨25481⟩⟩) (rawTerms := some (Proof.Events884.exact226522RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound226520.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 226526)
      LeftBound226526.bound (LeftBound226526.actual selector witness) := by
  exact .transfer (LeftBound226526.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound226520.bound LeftBound226526.bound
def bound : CoeffClass := .finite ⟨18743296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound226520.bound, LeftBound226526.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound226520.actual selector witness) * (LeftBound226526.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound226527

namespace LeftBound226533
def owner : Owner := ⟨.program ⟨257⟩, ⟨62442⟩⟩
def transferEvent : Nat := 226533
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 226531 .coefficient) (.predecessor 1 226532 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 226531 .coefficient)
      LeftAuthority10775.bound (LeftAuthority10775.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events042.exact10776RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10775.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10775.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 226532 .coefficient)
      LeftBound222151.bound (LeftBound222151.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events867.exact222153RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound222151.bound, RecordedBoundRefines] <;> decide)
      (LeftBound222151.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority10775.bound LeftBound222151.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10775.bound, LeftBound222151.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority10775.actual selector witness) * (LeftBound222151.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound226533

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
