import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard651
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard653
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard654
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard655
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard657
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard658
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard659
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard661
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard675

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound104721
def owner : Owner := ⟨.program ⟨257⟩, ⟨70563⟩⟩
def transferEvent : Nat := 104721
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 104719 .coefficient, .predecessor 1 104720 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 104719 .coefficient)
      LeftBound104716.bound (LeftBound104716.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events409.exact104718RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104716.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104716.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 104720 .coefficient)
      LeftBound102514.bound (LeftBound102514.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events400.exact102521RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102514.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102514.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104716.bound, LeftBound102514.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104716.bound, LeftBound102514.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104716.actual selector witness, LeftBound102514.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104721

namespace LeftBound104722
def owner : Owner := ⟨.program ⟨257⟩, ⟨70563⟩⟩
def transferEvent : Nat := 104722
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 104718 .summary, .result 102521 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 104718 .summary)
      LeftBound104717.bound (LeftBound104717.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70562⟩⟩) (rawTerms := some (Proof.Events409.exact104718RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104717.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 102521 .summary)
      LeftBound102516.bound (LeftBound102516.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28412⟩⟩) (rawTerms := some (Proof.Events400.exact102521RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound102516.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104717.bound, LeftBound102516.bound]
def bound : CoeffClass := .finite ⟨3802007596962448506045899439491360353157172, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104717.bound, LeftBound102516.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104717.actual selector witness, LeftBound102516.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104722

namespace LeftBound104726
def owner : Owner := ⟨.program ⟨257⟩, ⟨70564⟩⟩
def transferEvent : Nat := 104726
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 104724 .coefficient, .predecessor 1 104725 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 104724 .coefficient)
      LeftBound104721.bound (LeftBound104721.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events409.exact104723RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104721.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104721.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 104725 .coefficient)
      LeftBound102302.bound (LeftBound102302.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events399.exact102309RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102302.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102302.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104721.bound, LeftBound102302.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104721.bound, LeftBound102302.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104721.actual selector witness, LeftBound102302.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104726

namespace LeftBound104727
def owner : Owner := ⟨.program ⟨257⟩, ⟨70564⟩⟩
def transferEvent : Nat := 104727
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 104723 .summary, .result 102309 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 104723 .summary)
      LeftBound104722.bound (LeftBound104722.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70563⟩⟩) (rawTerms := some (Proof.Events409.exact104723RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104722.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 102309 .summary)
      LeftBound102304.bound (LeftBound102304.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨31092⟩⟩) (rawTerms := some (Proof.Events399.exact102309RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound102304.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104722.bound, LeftBound102304.bound]
def bound : CoeffClass := .finite ⟨4147668141949793872257454032897973461975092, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104722.bound, LeftBound102304.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104722.actual selector witness, LeftBound102304.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104727

namespace LeftBound104731
def owner : Owner := ⟨.program ⟨257⟩, ⟨70565⟩⟩
def transferEvent : Nat := 104731
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 104729 .coefficient, .predecessor 1 104730 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 104729 .coefficient)
      LeftBound104726.bound (LeftBound104726.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events409.exact104728RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104726.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104726.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 104730 .coefficient)
      LeftBound102090.bound (LeftBound102090.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events398.exact102097RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102090.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102090.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104726.bound, LeftBound102090.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104726.bound, LeftBound102090.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104726.actual selector witness, LeftBound102090.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104731

namespace LeftBound104732
def owner : Owner := ⟨.program ⟨257⟩, ⟨70565⟩⟩
def transferEvent : Nat := 104732
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 104728 .summary, .result 102097 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 104728 .summary)
      LeftBound104727.bound (LeftBound104727.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70564⟩⟩) (rawTerms := some (Proof.Events409.exact104728RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104727.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 102097 .summary)
      LeftBound102092.bound (LeftBound102092.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36752⟩⟩) (rawTerms := some (Proof.Events398.exact102097RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound102092.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104727.bound, LeftBound102092.bound]
def bound : CoeffClass := .finite ⟨4493332905678336798016456807332854062121012, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104727.bound, LeftBound102092.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104727.actual selector witness, LeftBound102092.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104732

namespace LeftBound104736
def owner : Owner := ⟨.program ⟨257⟩, ⟨70566⟩⟩
def transferEvent : Nat := 104736
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 104734 .coefficient, .predecessor 1 104735 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 104734 .coefficient)
      LeftBound104731.bound (LeftBound104731.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events409.exact104733RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104731.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104731.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 104735 .coefficient)
      LeftBound101878.bound (LeftBound101878.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events397.exact101885RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101878.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101878.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104731.bound, LeftBound101878.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104731.bound, LeftBound101878.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104731.actual selector witness, LeftBound101878.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104736

namespace LeftBound104737
def owner : Owner := ⟨.program ⟨257⟩, ⟨70566⟩⟩
def transferEvent : Nat := 104737
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 104733 .summary, .result 101885 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 104733 .summary)
      LeftBound104732.bound (LeftBound104732.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70565⟩⟩) (rawTerms := some (Proof.Events409.exact104733RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104732.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 101885 .summary)
      LeftBound101880.bound (LeftBound101880.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39432⟩⟩) (rawTerms := some (Proof.Events397.exact101885RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound101880.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104732.bound, LeftBound101880.bound]
def bound : CoeffClass := .finite ⟨4838999778777478503549183672281868407930932, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104732.bound, LeftBound101880.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104732.actual selector witness, LeftBound101880.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104737

namespace LeftBound104741
def owner : Owner := ⟨.program ⟨257⟩, ⟨70567⟩⟩
def transferEvent : Nat := 104741
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 104739 .coefficient, .predecessor 1 104740 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 104739 .coefficient)
      LeftBound104736.bound (LeftBound104736.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events409.exact104738RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104736.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104736.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 104740 .coefficient)
      LeftBound101666.bound (LeftBound101666.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events397.exact101673RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101666.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101666.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104736.bound, LeftBound101666.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104736.bound, LeftBound101666.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104736.actual selector witness, LeftBound101666.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104741

namespace LeftBound104742
def owner : Owner := ⟨.program ⟨257⟩, ⟨70567⟩⟩
def transferEvent : Nat := 104742
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 104738 .summary, .result 101673 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 104738 .summary)
      LeftBound104737.bound (LeftBound104737.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70566⟩⟩) (rawTerms := some (Proof.Events409.exact104738RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104737.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 101673 .summary)
      LeftBound101668.bound (LeftBound101668.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨42112⟩⟩) (rawTerms := some (Proof.Events397.exact101673RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound101668.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104737.bound, LeftBound101668.bound]
def bound : CoeffClass := .finite ⟨5184670870617817768629358718259150245068852, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104737.bound, LeftBound101668.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104737.actual selector witness, LeftBound101668.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104742

namespace LeftBound104746
def owner : Owner := ⟨.program ⟨257⟩, ⟨70568⟩⟩
def transferEvent : Nat := 104746
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 104744 .coefficient, .predecessor 1 104745 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 104744 .coefficient)
      LeftBound104741.bound (LeftBound104741.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events409.exact104743RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104741.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104741.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 104745 .coefficient)
      LeftBound101454.bound (LeftBound101454.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events396.exact101461RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101454.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101454.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104741.bound, LeftBound101454.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104741.bound, LeftBound101454.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104741.actual selector witness, LeftBound101454.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104746

namespace LeftBound104747
def owner : Owner := ⟨.program ⟨257⟩, ⟨70568⟩⟩
def transferEvent : Nat := 104747
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 104743 .summary, .result 101461 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 104743 .summary)
      LeftBound104742.bound (LeftBound104742.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70567⟩⟩) (rawTerms := some (Proof.Events409.exact104743RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104742.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 101461 .summary)
      LeftBound101456.bound (LeftBound101456.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44792⟩⟩) (rawTerms := some (Proof.Events396.exact101461RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound101456.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104742.bound, LeftBound101456.bound]
def bound : CoeffClass := .finite ⟨5530348290569953373030706035778833319198772, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104742.bound, LeftBound101456.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104742.actual selector witness, LeftBound101456.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104747

namespace LeftBound104751
def owner : Owner := ⟨.program ⟨257⟩, ⟨70569⟩⟩
def transferEvent : Nat := 104751
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 104749 .coefficient, .predecessor 1 104750 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 104749 .coefficient)
      LeftBound104746.bound (LeftBound104746.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events409.exact104748RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104746.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104746.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 104750 .coefficient)
      LeftBound101242.bound (LeftBound101242.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events395.exact101249RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101242.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101242.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104746.bound, LeftBound101242.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104746.bound, LeftBound101242.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104746.actual selector witness, LeftBound101242.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104751

namespace LeftBound104752
def owner : Owner := ⟨.program ⟨257⟩, ⟨70569⟩⟩
def transferEvent : Nat := 104752
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 104748 .summary, .result 101249 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 104748 .summary)
      LeftBound104747.bound (LeftBound104747.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70568⟩⟩) (rawTerms := some (Proof.Events409.exact104748RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104747.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 101249 .summary)
      LeftBound101244.bound (LeftBound101244.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨47472⟩⟩) (rawTerms := some (Proof.Events395.exact101249RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound101244.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104747.bound, LeftBound101244.bound]
def bound : CoeffClass := .finite ⟨5876032038633885316753225624840917630320692, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104747.bound, LeftBound101244.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104747.actual selector witness, LeftBound101244.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104752

namespace LeftBound104756
def owner : Owner := ⟨.program ⟨257⟩, ⟨70570⟩⟩
def transferEvent : Nat := 104756
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 104754 .coefficient, .predecessor 1 104755 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 104754 .coefficient)
      LeftBound104751.bound (LeftBound104751.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events409.exact104753RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104751.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104751.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 104755 .coefficient)
      LeftBound101030.bound (LeftBound101030.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events394.exact101037RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101030.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101030.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104751.bound, LeftBound101030.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104751.bound, LeftBound101030.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104751.actual selector witness, LeftBound101030.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104756

namespace LeftBound104757
def owner : Owner := ⟨.program ⟨257⟩, ⟨70570⟩⟩
def transferEvent : Nat := 104757
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 104753 .summary, .result 101037 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 104753 .summary)
      LeftBound104752.bound (LeftBound104752.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70569⟩⟩) (rawTerms := some (Proof.Events409.exact104753RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104752.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 101037 .summary)
      LeftBound101032.bound (LeftBound101032.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨50152⟩⟩) (rawTerms := some (Proof.Events394.exact101037RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound101032.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104752.bound, LeftBound101032.bound]
def bound : CoeffClass := .finite ⟨6221717896068416040249469304417135687106612, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104752.bound, LeftBound101032.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104752.actual selector witness, LeftBound101032.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104757

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
