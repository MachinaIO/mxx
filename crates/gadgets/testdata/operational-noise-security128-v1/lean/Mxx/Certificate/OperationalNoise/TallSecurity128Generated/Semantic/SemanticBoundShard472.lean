import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard460
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard462
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard463
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard464
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard466
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard467
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard468
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard469
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard470
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard471

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound75426
def owner : Owner := ⟨.program ⟨257⟩, ⟨20867⟩⟩
def transferEvent : Nat := 75426
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 75424 .coefficient, .predecessor 1 75425 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 75424 .coefficient)
      LeftBound75421.bound (LeftBound75421.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75423RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75421.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75421.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 75425 .coefficient)
      LeftBound75172.bound (LeftBound75172.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events293.exact75179RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75172.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75172.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75421.bound, LeftBound75172.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75421.bound, LeftBound75172.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound75421.actual selector witness, LeftBound75172.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75426

namespace LeftBound75427
def owner : Owner := ⟨.program ⟨257⟩, ⟨20867⟩⟩
def transferEvent : Nat := 75427
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 75423 .summary, .result 75179 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 75423 .summary)
      LeftBound75422.bound (LeftBound75422.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17955⟩⟩) (rawTerms := some (Proof.Events294.exact75423RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound75422.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 75179 .summary)
      LeftBound75174.bound (LeftBound75174.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20866⟩⟩) (rawTerms := some (Proof.Events293.exact75179RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound75174.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75422.bound, LeftBound75174.bound]
def bound : CoeffClass := .finite ⟨691250426059631610003352154589745737891892, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75422.bound, LeftBound75174.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound75422.actual selector witness, LeftBound75174.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75427

namespace LeftBound75431
def owner : Owner := ⟨.program ⟨257⟩, ⟨24087⟩⟩
def transferEvent : Nat := 75431
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 75429 .coefficient, .predecessor 1 75430 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 75429 .coefficient)
      LeftBound75426.bound (LeftBound75426.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75428RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75426.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75426.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 75430 .coefficient)
      LeftBound74960.bound (LeftBound74960.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events292.exact74967RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound74960.bound, RecordedBoundRefines] <;> decide)
      (LeftBound74960.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75426.bound, LeftBound74960.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75426.bound, LeftBound74960.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound75426.actual selector witness, LeftBound74960.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75431

namespace LeftBound75432
def owner : Owner := ⟨.program ⟨257⟩, ⟨24087⟩⟩
def transferEvent : Nat := 75432
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 75428 .summary, .result 74967 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 75428 .summary)
      LeftBound75427.bound (LeftBound75427.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20867⟩⟩) (rawTerms := some (Proof.Events294.exact75428RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound75427.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 74967 .summary)
      LeftBound74962.bound (LeftBound74962.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨24086⟩⟩) (rawTerms := some (Proof.Events292.exact74967RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound74962.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75427.bound, LeftBound74962.bound]
def bound : CoeffClass := .finite ⟨1036877221117396499835321299770218916085812, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75427.bound, LeftBound74962.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound75427.actual selector witness, LeftBound74962.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75432

namespace LeftBound75436
def owner : Owner := ⟨.program ⟨257⟩, ⟨34107⟩⟩
def transferEvent : Nat := 75436
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 75434 .coefficient, .predecessor 1 75435 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 75434 .coefficient)
      LeftBound75431.bound (LeftBound75431.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75433RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75431.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75431.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 75435 .coefficient)
      LeftBound74748.bound (LeftBound74748.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events292.exact74755RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound74748.bound, RecordedBoundRefines] <;> decide)
      (LeftBound74748.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75431.bound, LeftBound74748.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75431.bound, LeftBound74748.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound75431.actual selector witness, LeftBound74748.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75436

namespace LeftBound75437
def owner : Owner := ⟨.program ⟨257⟩, ⟨34107⟩⟩
def transferEvent : Nat := 75437
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 75433 .summary, .result 74755 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 75433 .summary)
      LeftBound75432.bound (LeftBound75432.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨24087⟩⟩) (rawTerms := some (Proof.Events294.exact75433RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound75432.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 74755 .summary)
      LeftBound74750.bound (LeftBound74750.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨34106⟩⟩) (rawTerms := some (Proof.Events292.exact74755RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound74750.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75432.bound, LeftBound74750.bound]
def bound : CoeffClass := .finite ⟨1382506125545760169441014535464825839943732, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75432.bound, LeftBound74750.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound75432.actual selector witness, LeftBound74750.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75437

namespace LeftBound75441
def owner : Owner := ⟨.program ⟨257⟩, ⟨53167⟩⟩
def transferEvent : Nat := 75441
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 75439 .coefficient, .predecessor 1 75440 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 75439 .coefficient)
      LeftBound75436.bound (LeftBound75436.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75438RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75436.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75436.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 75440 .coefficient)
      LeftBound74536.bound (LeftBound74536.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events291.exact74543RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound74536.bound, RecordedBoundRefines] <;> decide)
      (LeftBound74536.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75436.bound, LeftBound74536.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75436.bound, LeftBound74536.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound75436.actual selector witness, LeftBound74536.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75441

namespace LeftBound75442
def owner : Owner := ⟨.program ⟨257⟩, ⟨53167⟩⟩
def transferEvent : Nat := 75442
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 75438 .summary, .result 74543 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 75438 .summary)
      LeftBound75437.bound (LeftBound75437.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨34107⟩⟩) (rawTerms := some (Proof.Events294.exact75438RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound75437.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 74543 .summary)
      LeftBound74538.bound (LeftBound74538.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨53166⟩⟩) (rawTerms := some (Proof.Events291.exact74543RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound74538.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75437.bound, LeftBound74538.bound]
def bound : CoeffClass := .finite ⟨1728139248715321398594155952187700255129652, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75437.bound, LeftBound74538.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound75437.actual selector witness, LeftBound74538.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75442

namespace LeftBound75446
def owner : Owner := ⟨.program ⟨257⟩, ⟨56147⟩⟩
def transferEvent : Nat := 75446
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 75444 .coefficient, .predecessor 1 75445 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 75444 .coefficient)
      LeftBound75441.bound (LeftBound75441.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75443RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75441.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75441.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 75445 .coefficient)
      LeftBound74324.bound (LeftBound74324.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events290.exact74331RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound74324.bound, RecordedBoundRefines] <;> decide)
      (LeftBound74324.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75441.bound, LeftBound74324.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75441.bound, LeftBound74324.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound75441.actual selector witness, LeftBound74324.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75446

namespace LeftBound75447
def owner : Owner := ⟨.program ⟨257⟩, ⟨56147⟩⟩
def transferEvent : Nat := 75447
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 75443 .summary, .result 74331 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 75443 .summary)
      LeftBound75442.bound (LeftBound75442.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨53167⟩⟩) (rawTerms := some (Proof.Events294.exact75443RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound75442.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 74331 .summary)
      LeftBound74326.bound (LeftBound74326.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨56146⟩⟩) (rawTerms := some (Proof.Events290.exact74331RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound74326.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75442.bound, LeftBound74326.bound]
def bound : CoeffClass := .finite ⟨2073774481255481407521021459424708415979572, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75442.bound, LeftBound74326.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound75442.actual selector witness, LeftBound74326.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75447

namespace LeftBound75451
def owner : Owner := ⟨.program ⟨257⟩, ⟨59127⟩⟩
def transferEvent : Nat := 75451
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 75449 .coefficient, .predecessor 1 75450 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 75449 .coefficient)
      LeftBound75446.bound (LeftBound75446.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75448RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75446.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75446.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 75450 .coefficient)
      LeftBound74112.bound (LeftBound74112.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events289.exact74119RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound74112.bound, RecordedBoundRefines] <;> decide)
      (LeftBound74112.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75446.bound, LeftBound74112.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75446.bound, LeftBound74112.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound75446.actual selector witness, LeftBound74112.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75451

namespace LeftBound75452
def owner : Owner := ⟨.program ⟨257⟩, ⟨59127⟩⟩
def transferEvent : Nat := 75452
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 75448 .summary, .result 74119 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 75448 .summary)
      LeftBound75447.bound (LeftBound75447.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨56147⟩⟩) (rawTerms := some (Proof.Events294.exact75448RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound75447.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 74119 .summary)
      LeftBound74114.bound (LeftBound74114.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨59126⟩⟩) (rawTerms := some (Proof.Events289.exact74119RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound74114.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75447.bound, LeftBound74114.bound]
def bound : CoeffClass := .finite ⟨2419413932536838975995335147689984068157492, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75447.bound, LeftBound74114.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound75447.actual selector witness, LeftBound74114.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75452

namespace LeftBound75456
def owner : Owner := ⟨.program ⟨257⟩, ⟨62107⟩⟩
def transferEvent : Nat := 75456
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 75454 .coefficient, .predecessor 1 75455 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 75454 .coefficient)
      LeftBound75451.bound (LeftBound75451.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75453RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75451.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75451.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 75455 .coefficient)
      LeftBound73900.bound (LeftBound73900.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events288.exact73907RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73900.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73900.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75451.bound, LeftBound73900.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75451.bound, LeftBound73900.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound75451.actual selector witness, LeftBound73900.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75456

namespace LeftBound75457
def owner : Owner := ⟨.program ⟨257⟩, ⟨62107⟩⟩
def transferEvent : Nat := 75457
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 75453 .summary, .result 73907 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 75453 .summary)
      LeftBound75452.bound (LeftBound75452.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨59127⟩⟩) (rawTerms := some (Proof.Events294.exact75453RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound75452.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 73907 .summary)
      LeftBound73902.bound (LeftBound73902.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨62106⟩⟩) (rawTerms := some (Proof.Events288.exact73907RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound73902.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75452.bound, LeftBound73902.bound]
def bound : CoeffClass := .finite ⟨2765055493188795324243372926469393465999412, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75452.bound, LeftBound73902.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound75452.actual selector witness, LeftBound73902.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75457

namespace LeftBound75461
def owner : Owner := ⟨.program ⟨257⟩, ⟨65087⟩⟩
def transferEvent : Nat := 75461
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 75459 .coefficient, .predecessor 1 75460 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 75459 .coefficient)
      LeftBound75456.bound (LeftBound75456.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75458RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75456.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75456.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 75460 .coefficient)
      LeftBound73688.bound (LeftBound73688.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events287.exact73695RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73688.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73688.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75456.bound, LeftBound73688.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75456.bound, LeftBound73688.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound75456.actual selector witness, LeftBound73688.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75461

namespace LeftBound75462
def owner : Owner := ⟨.program ⟨257⟩, ⟨65087⟩⟩
def transferEvent : Nat := 75462
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 75458 .summary, .result 73695 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 75458 .summary)
      LeftBound75457.bound (LeftBound75457.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨62107⟩⟩) (rawTerms := some (Proof.Events294.exact75458RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound75457.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 73695 .summary)
      LeftBound73690.bound (LeftBound73690.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨65086⟩⟩) (rawTerms := some (Proof.Events287.exact73695RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound73690.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75457.bound, LeftBound73690.bound]
def bound : CoeffClass := .finite ⟨3110701272581949232038858886277070355169332, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75457.bound, LeftBound73690.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound75457.actual selector witness, LeftBound73690.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75462

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
