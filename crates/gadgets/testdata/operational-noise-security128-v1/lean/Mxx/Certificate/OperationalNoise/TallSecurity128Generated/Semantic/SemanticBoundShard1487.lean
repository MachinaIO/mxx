import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1468
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1469
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1470
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1471
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1472
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1473
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1475
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1476
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1477
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1486

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound221701
def owner : Owner := ⟨.program ⟨257⟩, ⟨58910⟩⟩
def transferEvent : Nat := 221701
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 221699 .coefficient, .predecessor 1 221700 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 221699 .coefficient)
      LeftBound221696.bound (LeftBound221696.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events866.exact221698RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound221696.bound, RecordedBoundRefines] <;> decide)
      (LeftBound221696.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 221700 .coefficient)
      LeftBound220362.bound (LeftBound220362.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events860.exact220369RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound220362.bound, RecordedBoundRefines] <;> decide)
      (LeftBound220362.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound221696.bound, LeftBound220362.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound221696.bound, LeftBound220362.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound221696.actual selector witness, LeftBound220362.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound221701

namespace LeftBound221702
def owner : Owner := ⟨.program ⟨257⟩, ⟨58910⟩⟩
def transferEvent : Nat := 221702
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 221698 .summary, .result 220369 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 221698 .summary)
      LeftBound221697.bound (LeftBound221697.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55930⟩⟩) (rawTerms := some (Proof.Events866.exact221698RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound221697.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 220369 .summary)
      LeftBound220364.bound (LeftBound220364.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58909⟩⟩) (rawTerms := some (Proof.Events860.exact220369RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound220364.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound221697.bound, LeftBound220364.bound]
def bound : CoeffClass := .finite ⟨2419413932536838975995335147689984068157492, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound221697.bound, LeftBound220364.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound221697.actual selector witness, LeftBound220364.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound221702

namespace LeftBound221706
def owner : Owner := ⟨.program ⟨257⟩, ⟨61890⟩⟩
def transferEvent : Nat := 221706
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 221704 .coefficient, .predecessor 1 221705 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 221704 .coefficient)
      LeftBound221701.bound (LeftBound221701.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events866.exact221703RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound221701.bound, RecordedBoundRefines] <;> decide)
      (LeftBound221701.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 221705 .coefficient)
      LeftBound220150.bound (LeftBound220150.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events859.exact220157RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound220150.bound, RecordedBoundRefines] <;> decide)
      (LeftBound220150.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound221701.bound, LeftBound220150.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound221701.bound, LeftBound220150.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound221701.actual selector witness, LeftBound220150.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound221706

namespace LeftBound221707
def owner : Owner := ⟨.program ⟨257⟩, ⟨61890⟩⟩
def transferEvent : Nat := 221707
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 221703 .summary, .result 220157 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 221703 .summary)
      LeftBound221702.bound (LeftBound221702.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58910⟩⟩) (rawTerms := some (Proof.Events866.exact221703RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound221702.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 220157 .summary)
      LeftBound220152.bound (LeftBound220152.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61889⟩⟩) (rawTerms := some (Proof.Events859.exact220157RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound220152.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound221702.bound, LeftBound220152.bound]
def bound : CoeffClass := .finite ⟨2765055493188795324243372926469393465999412, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound221702.bound, LeftBound220152.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound221702.actual selector witness, LeftBound220152.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound221707

namespace LeftBound221711
def owner : Owner := ⟨.program ⟨257⟩, ⟨64870⟩⟩
def transferEvent : Nat := 221711
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 221709 .coefficient, .predecessor 1 221710 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 221709 .coefficient)
      LeftBound221706.bound (LeftBound221706.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events866.exact221708RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound221706.bound, RecordedBoundRefines] <;> decide)
      (LeftBound221706.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 221710 .coefficient)
      LeftBound219938.bound (LeftBound219938.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events859.exact219945RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound219938.bound, RecordedBoundRefines] <;> decide)
      (LeftBound219938.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound221706.bound, LeftBound219938.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound221706.bound, LeftBound219938.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound221706.actual selector witness, LeftBound219938.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound221711

namespace LeftBound221712
def owner : Owner := ⟨.program ⟨257⟩, ⟨64870⟩⟩
def transferEvent : Nat := 221712
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 221708 .summary, .result 219945 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 221708 .summary)
      LeftBound221707.bound (LeftBound221707.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61890⟩⟩) (rawTerms := some (Proof.Events866.exact221708RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound221707.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 219945 .summary)
      LeftBound219940.bound (LeftBound219940.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64869⟩⟩) (rawTerms := some (Proof.Events859.exact219945RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound219940.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound221707.bound, LeftBound219940.bound]
def bound : CoeffClass := .finite ⟨3110701272581949232038858886277070355169332, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound221707.bound, LeftBound219940.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound221707.actual selector witness, LeftBound219940.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound221712

namespace LeftBound221716
def owner : Owner := ⟨.program ⟨257⟩, ⟨70167⟩⟩
def transferEvent : Nat := 221716
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 221714 .coefficient, .predecessor 1 221715 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 221714 .coefficient)
      LeftBound221711.bound (LeftBound221711.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events866.exact221713RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound221711.bound, RecordedBoundRefines] <;> decide)
      (LeftBound221711.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 221715 .coefficient)
      LeftBound219726.bound (LeftBound219726.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events858.exact219733RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound219726.bound, RecordedBoundRefines] <;> decide)
      (LeftBound219726.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound221711.bound, LeftBound219726.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound221711.bound, LeftBound219726.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound221711.actual selector witness, LeftBound219726.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound221716

namespace LeftBound221717
def owner : Owner := ⟨.program ⟨257⟩, ⟨70167⟩⟩
def transferEvent : Nat := 221717
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 221713 .summary, .result 219733 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 221713 .summary)
      LeftBound221712.bound (LeftBound221712.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64870⟩⟩) (rawTerms := some (Proof.Events866.exact221713RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound221712.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 219733 .summary)
      LeftBound219728.bound (LeftBound219728.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70166⟩⟩) (rawTerms := some (Proof.Events858.exact219733RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound219728.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound221712.bound, LeftBound219728.bound]
def bound : CoeffClass := .finite ⟨3456353380086899479155517117627148481331252, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound221712.bound, LeftBound219728.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound221712.actual selector witness, LeftBound219728.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound221717

namespace LeftBound221721
def owner : Owner := ⟨.program ⟨257⟩, ⟨70168⟩⟩
def transferEvent : Nat := 221721
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 221719 .coefficient, .predecessor 1 221720 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 221719 .coefficient)
      LeftBound221716.bound (LeftBound221716.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events866.exact221718RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound221716.bound, RecordedBoundRefines] <;> decide)
      (LeftBound221716.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 221720 .coefficient)
      LeftBound219514.bound (LeftBound219514.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events857.exact219521RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound219514.bound, RecordedBoundRefines] <;> decide)
      (LeftBound219514.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound221716.bound, LeftBound219514.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound221716.bound, LeftBound219514.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound221716.actual selector witness, LeftBound219514.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound221721

namespace LeftBound221722
def owner : Owner := ⟨.program ⟨257⟩, ⟨70168⟩⟩
def transferEvent : Nat := 221722
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 221718 .summary, .result 219521 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 221718 .summary)
      LeftBound221717.bound (LeftBound221717.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70167⟩⟩) (rawTerms := some (Proof.Events866.exact221718RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound221717.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 219521 .summary)
      LeftBound219516.bound (LeftBound219516.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28287⟩⟩) (rawTerms := some (Proof.Events857.exact219521RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound219516.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound221717.bound, LeftBound219516.bound]
def bound : CoeffClass := .finite ⟨3802007596962448506045899439491360353157172, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound221717.bound, LeftBound219516.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound221717.actual selector witness, LeftBound219516.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound221722

namespace LeftBound221726
def owner : Owner := ⟨.program ⟨257⟩, ⟨70169⟩⟩
def transferEvent : Nat := 221726
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 221724 .coefficient, .predecessor 1 221725 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 221724 .coefficient)
      LeftBound221721.bound (LeftBound221721.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events866.exact221723RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound221721.bound, RecordedBoundRefines] <;> decide)
      (LeftBound221721.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 221725 .coefficient)
      LeftBound219302.bound (LeftBound219302.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events856.exact219309RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound219302.bound, RecordedBoundRefines] <;> decide)
      (LeftBound219302.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound221721.bound, LeftBound219302.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound221721.bound, LeftBound219302.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound221721.actual selector witness, LeftBound219302.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound221726

namespace LeftBound221727
def owner : Owner := ⟨.program ⟨257⟩, ⟨70169⟩⟩
def transferEvent : Nat := 221727
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 221723 .summary, .result 219309 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 221723 .summary)
      LeftBound221722.bound (LeftBound221722.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70168⟩⟩) (rawTerms := some (Proof.Events866.exact221723RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound221722.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 219309 .summary)
      LeftBound219304.bound (LeftBound219304.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨30967⟩⟩) (rawTerms := some (Proof.Events856.exact219309RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound219304.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound221722.bound, LeftBound219304.bound]
def bound : CoeffClass := .finite ⟨4147668141949793872257454032897973461975092, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound221722.bound, LeftBound219304.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound221722.actual selector witness, LeftBound219304.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound221727

namespace LeftBound221731
def owner : Owner := ⟨.program ⟨257⟩, ⟨70170⟩⟩
def transferEvent : Nat := 221731
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 221729 .coefficient, .predecessor 1 221730 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 221729 .coefficient)
      LeftBound221726.bound (LeftBound221726.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events866.exact221728RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound221726.bound, RecordedBoundRefines] <;> decide)
      (LeftBound221726.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 221730 .coefficient)
      LeftBound219090.bound (LeftBound219090.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events855.exact219097RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound219090.bound, RecordedBoundRefines] <;> decide)
      (LeftBound219090.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound221726.bound, LeftBound219090.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound221726.bound, LeftBound219090.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound221726.actual selector witness, LeftBound219090.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound221731

namespace LeftBound221732
def owner : Owner := ⟨.program ⟨257⟩, ⟨70170⟩⟩
def transferEvent : Nat := 221732
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 221728 .summary, .result 219097 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 221728 .summary)
      LeftBound221727.bound (LeftBound221727.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70169⟩⟩) (rawTerms := some (Proof.Events866.exact221728RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound221727.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 219097 .summary)
      LeftBound219092.bound (LeftBound219092.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36627⟩⟩) (rawTerms := some (Proof.Events855.exact219097RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound219092.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound221727.bound, LeftBound219092.bound]
def bound : CoeffClass := .finite ⟨4493332905678336798016456807332854062121012, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound221727.bound, LeftBound219092.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound221727.actual selector witness, LeftBound219092.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound221732

namespace LeftBound221736
def owner : Owner := ⟨.program ⟨257⟩, ⟨70171⟩⟩
def transferEvent : Nat := 221736
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 221734 .coefficient, .predecessor 1 221735 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 221734 .coefficient)
      LeftBound221731.bound (LeftBound221731.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events866.exact221733RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound221731.bound, RecordedBoundRefines] <;> decide)
      (LeftBound221731.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 221735 .coefficient)
      LeftBound218878.bound (LeftBound218878.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events855.exact218885RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound218878.bound, RecordedBoundRefines] <;> decide)
      (LeftBound218878.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound221731.bound, LeftBound218878.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound221731.bound, LeftBound218878.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound221731.actual selector witness, LeftBound218878.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound221736

namespace LeftBound221737
def owner : Owner := ⟨.program ⟨257⟩, ⟨70171⟩⟩
def transferEvent : Nat := 221737
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 221733 .summary, .result 218885 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 221733 .summary)
      LeftBound221732.bound (LeftBound221732.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70170⟩⟩) (rawTerms := some (Proof.Events866.exact221733RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound221732.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 218885 .summary)
      LeftBound218880.bound (LeftBound218880.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39307⟩⟩) (rawTerms := some (Proof.Events855.exact218885RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound218880.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound221732.bound, LeftBound218880.bound]
def bound : CoeffClass := .finite ⟨4838999778777478503549183672281868407930932, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound221732.bound, LeftBound218880.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound221732.actual selector witness, LeftBound218880.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound221737

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
