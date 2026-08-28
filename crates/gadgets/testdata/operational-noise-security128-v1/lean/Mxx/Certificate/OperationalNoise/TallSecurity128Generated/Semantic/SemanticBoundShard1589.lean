import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1563
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1564
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1566
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1567
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1568
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1570
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1571
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1572
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1574
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1588

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound236347
def owner : Owner := ⟨.program ⟨257⟩, ⟨70089⟩⟩
def transferEvent : Nat := 236347
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 236343 .summary, .result 234146 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 236343 .summary)
      LeftBound236342.bound (LeftBound236342.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70088⟩⟩) (rawTerms := some (Proof.Events923.exact236343RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound236342.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 234146 .summary)
      LeftBound234141.bound (LeftBound234141.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28262⟩⟩) (rawTerms := some (Proof.Events914.exact234146RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound234141.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound236342.bound, LeftBound234141.bound]
def bound : CoeffClass := .finite ⟨3802007596962448506045899439491360353157172, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236342.bound, LeftBound234141.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound236342.actual selector witness, LeftBound234141.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound236347

namespace LeftBound236351
def owner : Owner := ⟨.program ⟨257⟩, ⟨70090⟩⟩
def transferEvent : Nat := 236351
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 236349 .coefficient, .predecessor 1 236350 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 236349 .coefficient)
      LeftBound236346.bound (LeftBound236346.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events923.exact236348RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236346.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236346.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 236350 .coefficient)
      LeftBound233927.bound (LeftBound233927.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events913.exact233934RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound233927.bound, RecordedBoundRefines] <;> decide)
      (LeftBound233927.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound236346.bound, LeftBound233927.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236346.bound, LeftBound233927.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound236346.actual selector witness, LeftBound233927.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound236351

namespace LeftBound236352
def owner : Owner := ⟨.program ⟨257⟩, ⟨70090⟩⟩
def transferEvent : Nat := 236352
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 236348 .summary, .result 233934 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 236348 .summary)
      LeftBound236347.bound (LeftBound236347.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70089⟩⟩) (rawTerms := some (Proof.Events923.exact236348RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound236347.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 233934 .summary)
      LeftBound233929.bound (LeftBound233929.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨30942⟩⟩) (rawTerms := some (Proof.Events913.exact233934RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound233929.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound236347.bound, LeftBound233929.bound]
def bound : CoeffClass := .finite ⟨4147668141949793872257454032897973461975092, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236347.bound, LeftBound233929.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound236347.actual selector witness, LeftBound233929.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound236352

namespace LeftBound236356
def owner : Owner := ⟨.program ⟨257⟩, ⟨70091⟩⟩
def transferEvent : Nat := 236356
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 236354 .coefficient, .predecessor 1 236355 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 236354 .coefficient)
      LeftBound236351.bound (LeftBound236351.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events923.exact236353RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236351.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236351.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 236355 .coefficient)
      LeftBound233715.bound (LeftBound233715.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events912.exact233722RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound233715.bound, RecordedBoundRefines] <;> decide)
      (LeftBound233715.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound236351.bound, LeftBound233715.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236351.bound, LeftBound233715.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound236351.actual selector witness, LeftBound233715.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound236356

namespace LeftBound236357
def owner : Owner := ⟨.program ⟨257⟩, ⟨70091⟩⟩
def transferEvent : Nat := 236357
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 236353 .summary, .result 233722 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 236353 .summary)
      LeftBound236352.bound (LeftBound236352.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70090⟩⟩) (rawTerms := some (Proof.Events923.exact236353RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound236352.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 233722 .summary)
      LeftBound233717.bound (LeftBound233717.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36602⟩⟩) (rawTerms := some (Proof.Events912.exact233722RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound233717.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound236352.bound, LeftBound233717.bound]
def bound : CoeffClass := .finite ⟨4493332905678336798016456807332854062121012, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236352.bound, LeftBound233717.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound236352.actual selector witness, LeftBound233717.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound236357

namespace LeftBound236361
def owner : Owner := ⟨.program ⟨257⟩, ⟨70092⟩⟩
def transferEvent : Nat := 236361
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 236359 .coefficient, .predecessor 1 236360 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 236359 .coefficient)
      LeftBound236356.bound (LeftBound236356.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events923.exact236358RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236356.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236356.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 236360 .coefficient)
      LeftBound233503.bound (LeftBound233503.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events912.exact233510RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound233503.bound, RecordedBoundRefines] <;> decide)
      (LeftBound233503.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound236356.bound, LeftBound233503.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236356.bound, LeftBound233503.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound236356.actual selector witness, LeftBound233503.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound236361

namespace LeftBound236362
def owner : Owner := ⟨.program ⟨257⟩, ⟨70092⟩⟩
def transferEvent : Nat := 236362
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 236358 .summary, .result 233510 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 236358 .summary)
      LeftBound236357.bound (LeftBound236357.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70091⟩⟩) (rawTerms := some (Proof.Events923.exact236358RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound236357.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 233510 .summary)
      LeftBound233505.bound (LeftBound233505.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39282⟩⟩) (rawTerms := some (Proof.Events912.exact233510RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound233505.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound236357.bound, LeftBound233505.bound]
def bound : CoeffClass := .finite ⟨4838999778777478503549183672281868407930932, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236357.bound, LeftBound233505.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound236357.actual selector witness, LeftBound233505.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound236362

namespace LeftBound236366
def owner : Owner := ⟨.program ⟨257⟩, ⟨70093⟩⟩
def transferEvent : Nat := 236366
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 236364 .coefficient, .predecessor 1 236365 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 236364 .coefficient)
      LeftBound236361.bound (LeftBound236361.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events923.exact236363RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236361.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236361.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 236365 .coefficient)
      LeftBound233291.bound (LeftBound233291.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events911.exact233298RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound233291.bound, RecordedBoundRefines] <;> decide)
      (LeftBound233291.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound236361.bound, LeftBound233291.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236361.bound, LeftBound233291.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound236361.actual selector witness, LeftBound233291.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound236366

namespace LeftBound236367
def owner : Owner := ⟨.program ⟨257⟩, ⟨70093⟩⟩
def transferEvent : Nat := 236367
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 236363 .summary, .result 233298 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 236363 .summary)
      LeftBound236362.bound (LeftBound236362.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70092⟩⟩) (rawTerms := some (Proof.Events923.exact236363RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound236362.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 233298 .summary)
      LeftBound233293.bound (LeftBound233293.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨41962⟩⟩) (rawTerms := some (Proof.Events911.exact233298RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound233293.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound236362.bound, LeftBound233293.bound]
def bound : CoeffClass := .finite ⟨5184670870617817768629358718259150245068852, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236362.bound, LeftBound233293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound236362.actual selector witness, LeftBound233293.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound236367

namespace LeftBound236371
def owner : Owner := ⟨.program ⟨257⟩, ⟨70094⟩⟩
def transferEvent : Nat := 236371
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 236369 .coefficient, .predecessor 1 236370 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 236369 .coefficient)
      LeftBound236366.bound (LeftBound236366.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events923.exact236368RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236366.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236366.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 236370 .coefficient)
      LeftBound233079.bound (LeftBound233079.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events910.exact233086RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound233079.bound, RecordedBoundRefines] <;> decide)
      (LeftBound233079.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound236366.bound, LeftBound233079.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236366.bound, LeftBound233079.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound236366.actual selector witness, LeftBound233079.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound236371

namespace LeftBound236372
def owner : Owner := ⟨.program ⟨257⟩, ⟨70094⟩⟩
def transferEvent : Nat := 236372
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 236368 .summary, .result 233086 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 236368 .summary)
      LeftBound236367.bound (LeftBound236367.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70093⟩⟩) (rawTerms := some (Proof.Events923.exact236368RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound236367.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 233086 .summary)
      LeftBound233081.bound (LeftBound233081.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44642⟩⟩) (rawTerms := some (Proof.Events910.exact233086RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound233081.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound236367.bound, LeftBound233081.bound]
def bound : CoeffClass := .finite ⟨5530348290569953373030706035778833319198772, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236367.bound, LeftBound233081.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound236367.actual selector witness, LeftBound233081.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound236372

namespace LeftBound236376
def owner : Owner := ⟨.program ⟨257⟩, ⟨70095⟩⟩
def transferEvent : Nat := 236376
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 236374 .coefficient, .predecessor 1 236375 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 236374 .coefficient)
      LeftBound236371.bound (LeftBound236371.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events923.exact236373RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236371.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236371.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 236375 .coefficient)
      LeftBound232867.bound (LeftBound232867.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events909.exact232874RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound232867.bound, RecordedBoundRefines] <;> decide)
      (LeftBound232867.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound236371.bound, LeftBound232867.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236371.bound, LeftBound232867.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound236371.actual selector witness, LeftBound232867.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound236376

namespace LeftBound236377
def owner : Owner := ⟨.program ⟨257⟩, ⟨70095⟩⟩
def transferEvent : Nat := 236377
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 236373 .summary, .result 232874 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 236373 .summary)
      LeftBound236372.bound (LeftBound236372.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70094⟩⟩) (rawTerms := some (Proof.Events923.exact236373RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound236372.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 232874 .summary)
      LeftBound232869.bound (LeftBound232869.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨47322⟩⟩) (rawTerms := some (Proof.Events909.exact232874RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound232869.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound236372.bound, LeftBound232869.bound]
def bound : CoeffClass := .finite ⟨5876032038633885316753225624840917630320692, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236372.bound, LeftBound232869.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound236372.actual selector witness, LeftBound232869.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound236377

namespace LeftBound236381
def owner : Owner := ⟨.program ⟨257⟩, ⟨70096⟩⟩
def transferEvent : Nat := 236381
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 236379 .coefficient, .predecessor 1 236380 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 236379 .coefficient)
      LeftBound236376.bound (LeftBound236376.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events923.exact236378RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236376.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236376.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 236380 .coefficient)
      LeftBound232655.bound (LeftBound232655.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events908.exact232662RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound232655.bound, RecordedBoundRefines] <;> decide)
      (LeftBound232655.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound236376.bound, LeftBound232655.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236376.bound, LeftBound232655.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound236376.actual selector witness, LeftBound232655.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound236381

namespace LeftBound236382
def owner : Owner := ⟨.program ⟨257⟩, ⟨70096⟩⟩
def transferEvent : Nat := 236382
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 236378 .summary, .result 232662 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 236378 .summary)
      LeftBound236377.bound (LeftBound236377.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70095⟩⟩) (rawTerms := some (Proof.Events923.exact236378RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound236377.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 232662 .summary)
      LeftBound232657.bound (LeftBound232657.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨50002⟩⟩) (rawTerms := some (Proof.Events908.exact232662RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound232657.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound236377.bound, LeftBound232657.bound]
def bound : CoeffClass := .finite ⟨6221717896068416040249469304417135687106612, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236377.bound, LeftBound232657.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound236377.actual selector witness, LeftBound232657.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound236382

namespace LeftBound236386
def owner : Owner := ⟨.program ⟨257⟩, ⟨71210⟩⟩
def transferEvent : Nat := 236386
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 236384 .coefficient, .predecessor 1 236385 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 236384 .coefficient)
      LeftBound236381.bound (LeftBound236381.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events923.exact236383RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236381.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236381.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 236385 .coefficient)
      LeftBound232443.bound (LeftBound232443.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events908.exact232450RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound232443.bound, RecordedBoundRefines] <;> decide)
      (LeftBound232443.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound236381.bound, LeftBound232443.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236381.bound, LeftBound232443.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound236381.actual selector witness, LeftBound232443.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound236386

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
