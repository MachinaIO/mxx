import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard450
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard451
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard452
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard453
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard454
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard455
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard456
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard458
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard459
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard472

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound75466
def owner : Owner := ⟨.program ⟨257⟩, ⟨70720⟩⟩
def transferEvent : Nat := 75466
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 75464 .coefficient, .predecessor 1 75465 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 75464 .coefficient)
      LeftBound75461.bound (LeftBound75461.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75463RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75461.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75461.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 75465 .coefficient)
      LeftBound73476.bound (LeftBound73476.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events287.exact73483RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73476.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73476.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75461.bound, LeftBound73476.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75461.bound, LeftBound73476.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound75461.actual selector witness, LeftBound73476.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75466

namespace LeftBound75467
def owner : Owner := ⟨.program ⟨257⟩, ⟨70720⟩⟩
def transferEvent : Nat := 75467
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 75463 .summary, .result 73483 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 75463 .summary)
      LeftBound75462.bound (LeftBound75462.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨65087⟩⟩) (rawTerms := some (Proof.Events294.exact75463RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound75462.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 73483 .summary)
      LeftBound73478.bound (LeftBound73478.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70719⟩⟩) (rawTerms := some (Proof.Events287.exact73483RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound73478.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75462.bound, LeftBound73478.bound]
def bound : CoeffClass := .finite ⟨3456353380086899479155517117627148481331252, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75462.bound, LeftBound73478.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound75462.actual selector witness, LeftBound73478.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75467

namespace LeftBound75471
def owner : Owner := ⟨.program ⟨257⟩, ⟨70721⟩⟩
def transferEvent : Nat := 75471
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 75469 .coefficient, .predecessor 1 75470 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 75469 .coefficient)
      LeftBound75466.bound (LeftBound75466.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75468RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75466.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75466.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 75470 .coefficient)
      LeftBound73264.bound (LeftBound73264.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events286.exact73271RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73264.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73264.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75466.bound, LeftBound73264.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75466.bound, LeftBound73264.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound75466.actual selector witness, LeftBound73264.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75471

namespace LeftBound75472
def owner : Owner := ⟨.program ⟨257⟩, ⟨70721⟩⟩
def transferEvent : Nat := 75472
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 75468 .summary, .result 73271 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 75468 .summary)
      LeftBound75467.bound (LeftBound75467.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70720⟩⟩) (rawTerms := some (Proof.Events294.exact75468RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound75467.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 73271 .summary)
      LeftBound73266.bound (LeftBound73266.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28462⟩⟩) (rawTerms := some (Proof.Events286.exact73271RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound73266.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75467.bound, LeftBound73266.bound]
def bound : CoeffClass := .finite ⟨3802007596962448506045899439491360353157172, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75467.bound, LeftBound73266.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound75467.actual selector witness, LeftBound73266.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75472

namespace LeftBound75476
def owner : Owner := ⟨.program ⟨257⟩, ⟨70722⟩⟩
def transferEvent : Nat := 75476
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 75474 .coefficient, .predecessor 1 75475 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 75474 .coefficient)
      LeftBound75471.bound (LeftBound75471.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75473RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75471.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75471.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 75475 .coefficient)
      LeftBound73052.bound (LeftBound73052.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events285.exact73059RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73052.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73052.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75471.bound, LeftBound73052.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75471.bound, LeftBound73052.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound75471.actual selector witness, LeftBound73052.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75476

namespace LeftBound75477
def owner : Owner := ⟨.program ⟨257⟩, ⟨70722⟩⟩
def transferEvent : Nat := 75477
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 75473 .summary, .result 73059 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 75473 .summary)
      LeftBound75472.bound (LeftBound75472.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70721⟩⟩) (rawTerms := some (Proof.Events294.exact75473RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound75472.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 73059 .summary)
      LeftBound73054.bound (LeftBound73054.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨31142⟩⟩) (rawTerms := some (Proof.Events285.exact73059RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound73054.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75472.bound, LeftBound73054.bound]
def bound : CoeffClass := .finite ⟨4147668141949793872257454032897973461975092, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75472.bound, LeftBound73054.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound75472.actual selector witness, LeftBound73054.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75477

namespace LeftBound75481
def owner : Owner := ⟨.program ⟨257⟩, ⟨70723⟩⟩
def transferEvent : Nat := 75481
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 75479 .coefficient, .predecessor 1 75480 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 75479 .coefficient)
      LeftBound75476.bound (LeftBound75476.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75478RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75476.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75476.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 75480 .coefficient)
      LeftBound72840.bound (LeftBound72840.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events284.exact72847RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72840.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72840.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75476.bound, LeftBound72840.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75476.bound, LeftBound72840.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound75476.actual selector witness, LeftBound72840.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75481

namespace LeftBound75482
def owner : Owner := ⟨.program ⟨257⟩, ⟨70723⟩⟩
def transferEvent : Nat := 75482
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 75478 .summary, .result 72847 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 75478 .summary)
      LeftBound75477.bound (LeftBound75477.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70722⟩⟩) (rawTerms := some (Proof.Events294.exact75478RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound75477.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 72847 .summary)
      LeftBound72842.bound (LeftBound72842.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36802⟩⟩) (rawTerms := some (Proof.Events284.exact72847RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound72842.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75477.bound, LeftBound72842.bound]
def bound : CoeffClass := .finite ⟨4493332905678336798016456807332854062121012, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75477.bound, LeftBound72842.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound75477.actual selector witness, LeftBound72842.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75482

namespace LeftBound75486
def owner : Owner := ⟨.program ⟨257⟩, ⟨70724⟩⟩
def transferEvent : Nat := 75486
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 75484 .coefficient, .predecessor 1 75485 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 75484 .coefficient)
      LeftBound75481.bound (LeftBound75481.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75483RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75481.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75481.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 75485 .coefficient)
      LeftBound72628.bound (LeftBound72628.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events283.exact72635RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72628.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72628.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75481.bound, LeftBound72628.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75481.bound, LeftBound72628.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound75481.actual selector witness, LeftBound72628.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75486

namespace LeftBound75487
def owner : Owner := ⟨.program ⟨257⟩, ⟨70724⟩⟩
def transferEvent : Nat := 75487
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 75483 .summary, .result 72635 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 75483 .summary)
      LeftBound75482.bound (LeftBound75482.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70723⟩⟩) (rawTerms := some (Proof.Events294.exact75483RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound75482.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 72635 .summary)
      LeftBound72630.bound (LeftBound72630.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39482⟩⟩) (rawTerms := some (Proof.Events283.exact72635RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound72630.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75482.bound, LeftBound72630.bound]
def bound : CoeffClass := .finite ⟨4838999778777478503549183672281868407930932, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75482.bound, LeftBound72630.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound75482.actual selector witness, LeftBound72630.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75487

namespace LeftBound75491
def owner : Owner := ⟨.program ⟨257⟩, ⟨70725⟩⟩
def transferEvent : Nat := 75491
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 75489 .coefficient, .predecessor 1 75490 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 75489 .coefficient)
      LeftBound75486.bound (LeftBound75486.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75488RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75486.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75486.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 75490 .coefficient)
      LeftBound72416.bound (LeftBound72416.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events282.exact72423RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72416.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72416.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75486.bound, LeftBound72416.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75486.bound, LeftBound72416.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound75486.actual selector witness, LeftBound72416.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75491

namespace LeftBound75492
def owner : Owner := ⟨.program ⟨257⟩, ⟨70725⟩⟩
def transferEvent : Nat := 75492
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 75488 .summary, .result 72423 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 75488 .summary)
      LeftBound75487.bound (LeftBound75487.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70724⟩⟩) (rawTerms := some (Proof.Events294.exact75488RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound75487.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 72423 .summary)
      LeftBound72418.bound (LeftBound72418.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨42162⟩⟩) (rawTerms := some (Proof.Events282.exact72423RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound72418.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75487.bound, LeftBound72418.bound]
def bound : CoeffClass := .finite ⟨5184670870617817768629358718259150245068852, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75487.bound, LeftBound72418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound75487.actual selector witness, LeftBound72418.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75492

namespace LeftBound75496
def owner : Owner := ⟨.program ⟨257⟩, ⟨70726⟩⟩
def transferEvent : Nat := 75496
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 75494 .coefficient, .predecessor 1 75495 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 75494 .coefficient)
      LeftBound75491.bound (LeftBound75491.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75493RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75491.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75491.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 75495 .coefficient)
      LeftBound72204.bound (LeftBound72204.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events282.exact72211RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72204.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72204.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75491.bound, LeftBound72204.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75491.bound, LeftBound72204.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound75491.actual selector witness, LeftBound72204.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75496

namespace LeftBound75497
def owner : Owner := ⟨.program ⟨257⟩, ⟨70726⟩⟩
def transferEvent : Nat := 75497
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 75493 .summary, .result 72211 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 75493 .summary)
      LeftBound75492.bound (LeftBound75492.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70725⟩⟩) (rawTerms := some (Proof.Events294.exact75493RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound75492.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 72211 .summary)
      LeftBound72206.bound (LeftBound72206.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44842⟩⟩) (rawTerms := some (Proof.Events282.exact72211RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound72206.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75492.bound, LeftBound72206.bound]
def bound : CoeffClass := .finite ⟨5530348290569953373030706035778833319198772, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75492.bound, LeftBound72206.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound75492.actual selector witness, LeftBound72206.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75497

namespace LeftBound75501
def owner : Owner := ⟨.program ⟨257⟩, ⟨70727⟩⟩
def transferEvent : Nat := 75501
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 75499 .coefficient, .predecessor 1 75500 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 75499 .coefficient)
      LeftBound75496.bound (LeftBound75496.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75498RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75496.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75496.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 75500 .coefficient)
      LeftBound71992.bound (LeftBound71992.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact71999RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71992.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71992.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75496.bound, LeftBound71992.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75496.bound, LeftBound71992.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound75496.actual selector witness, LeftBound71992.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75501

namespace LeftBound75502
def owner : Owner := ⟨.program ⟨257⟩, ⟨70727⟩⟩
def transferEvent : Nat := 75502
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 75498 .summary, .result 71999 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 75498 .summary)
      LeftBound75497.bound (LeftBound75497.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70726⟩⟩) (rawTerms := some (Proof.Events294.exact75498RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound75497.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 71999 .summary)
      LeftBound71994.bound (LeftBound71994.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨47522⟩⟩) (rawTerms := some (Proof.Events281.exact71999RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound71994.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75497.bound, LeftBound71994.bound]
def bound : CoeffClass := .finite ⟨5876032038633885316753225624840917630320692, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75497.bound, LeftBound71994.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound75497.actual selector witness, LeftBound71994.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75502

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
