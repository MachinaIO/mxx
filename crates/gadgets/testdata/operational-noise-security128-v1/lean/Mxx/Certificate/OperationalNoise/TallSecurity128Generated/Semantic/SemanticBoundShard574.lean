import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard555
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard556
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard557
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard558
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard559
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard560
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard561
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard562
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard563
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard564
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard566
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard573

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound90072
def owner : Owner := ⟨.program ⟨257⟩, ⟨56116⟩⟩
def transferEvent : Nat := 90072
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 90068 .summary, .result 88956 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 90068 .summary)
      LeftBound90067.bound (LeftBound90067.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨53136⟩⟩) (rawTerms := some (Proof.Events351.exact90068RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound90067.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 88956 .summary)
      LeftBound88951.bound (LeftBound88951.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨56115⟩⟩) (rawTerms := some (Proof.Events347.exact88956RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound88951.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound90067.bound, LeftBound88951.bound]
def bound : CoeffClass := .finite ⟨2073774481255481407521021459424708415979572, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90067.bound, LeftBound88951.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound90067.actual selector witness, LeftBound88951.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90072

namespace LeftBound90076
def owner : Owner := ⟨.program ⟨257⟩, ⟨59096⟩⟩
def transferEvent : Nat := 90076
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 90074 .coefficient, .predecessor 1 90075 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 90074 .coefficient)
      LeftBound90071.bound (LeftBound90071.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact90073RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90071.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90071.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 90075 .coefficient)
      LeftBound88737.bound (LeftBound88737.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events346.exact88744RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88737.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88737.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound90071.bound, LeftBound88737.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90071.bound, LeftBound88737.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound90071.actual selector witness, LeftBound88737.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90076

namespace LeftBound90077
def owner : Owner := ⟨.program ⟨257⟩, ⟨59096⟩⟩
def transferEvent : Nat := 90077
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 90073 .summary, .result 88744 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 90073 .summary)
      LeftBound90072.bound (LeftBound90072.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨56116⟩⟩) (rawTerms := some (Proof.Events351.exact90073RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound90072.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 88744 .summary)
      LeftBound88739.bound (LeftBound88739.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨59095⟩⟩) (rawTerms := some (Proof.Events346.exact88744RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound88739.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound90072.bound, LeftBound88739.bound]
def bound : CoeffClass := .finite ⟨2419413932536838975995335147689984068157492, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90072.bound, LeftBound88739.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound90072.actual selector witness, LeftBound88739.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90077

namespace LeftBound90081
def owner : Owner := ⟨.program ⟨257⟩, ⟨62076⟩⟩
def transferEvent : Nat := 90081
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 90079 .coefficient, .predecessor 1 90080 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 90079 .coefficient)
      LeftBound90076.bound (LeftBound90076.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact90078RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90076.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90076.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 90080 .coefficient)
      LeftBound88525.bound (LeftBound88525.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events345.exact88532RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88525.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88525.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound90076.bound, LeftBound88525.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90076.bound, LeftBound88525.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound90076.actual selector witness, LeftBound88525.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90081

namespace LeftBound90082
def owner : Owner := ⟨.program ⟨257⟩, ⟨62076⟩⟩
def transferEvent : Nat := 90082
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 90078 .summary, .result 88532 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 90078 .summary)
      LeftBound90077.bound (LeftBound90077.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨59096⟩⟩) (rawTerms := some (Proof.Events351.exact90078RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound90077.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 88532 .summary)
      LeftBound88527.bound (LeftBound88527.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨62075⟩⟩) (rawTerms := some (Proof.Events345.exact88532RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound88527.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound90077.bound, LeftBound88527.bound]
def bound : CoeffClass := .finite ⟨2765055493188795324243372926469393465999412, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90077.bound, LeftBound88527.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound90077.actual selector witness, LeftBound88527.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90082

namespace LeftBound90086
def owner : Owner := ⟨.program ⟨257⟩, ⟨65056⟩⟩
def transferEvent : Nat := 90086
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 90084 .coefficient, .predecessor 1 90085 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 90084 .coefficient)
      LeftBound90081.bound (LeftBound90081.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact90083RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90081.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90081.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 90085 .coefficient)
      LeftBound88313.bound (LeftBound88313.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events345.exact88320RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88313.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88313.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound90081.bound, LeftBound88313.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90081.bound, LeftBound88313.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound90081.actual selector witness, LeftBound88313.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90086

namespace LeftBound90087
def owner : Owner := ⟨.program ⟨257⟩, ⟨65056⟩⟩
def transferEvent : Nat := 90087
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 90083 .summary, .result 88320 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 90083 .summary)
      LeftBound90082.bound (LeftBound90082.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨62076⟩⟩) (rawTerms := some (Proof.Events351.exact90083RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound90082.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 88320 .summary)
      LeftBound88315.bound (LeftBound88315.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨65055⟩⟩) (rawTerms := some (Proof.Events345.exact88320RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound88315.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound90082.bound, LeftBound88315.bound]
def bound : CoeffClass := .finite ⟨3110701272581949232038858886277070355169332, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90082.bound, LeftBound88315.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound90082.actual selector witness, LeftBound88315.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90087

namespace LeftBound90091
def owner : Owner := ⟨.program ⟨257⟩, ⟨70641⟩⟩
def transferEvent : Nat := 90091
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 90089 .coefficient, .predecessor 1 90090 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 90089 .coefficient)
      LeftBound90086.bound (LeftBound90086.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact90088RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90086.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90086.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 90090 .coefficient)
      LeftBound88101.bound (LeftBound88101.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events344.exact88108RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88101.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88101.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound90086.bound, LeftBound88101.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90086.bound, LeftBound88101.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound90086.actual selector witness, LeftBound88101.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90091

namespace LeftBound90092
def owner : Owner := ⟨.program ⟨257⟩, ⟨70641⟩⟩
def transferEvent : Nat := 90092
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 90088 .summary, .result 88108 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 90088 .summary)
      LeftBound90087.bound (LeftBound90087.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨65056⟩⟩) (rawTerms := some (Proof.Events351.exact90088RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound90087.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 88108 .summary)
      LeftBound88103.bound (LeftBound88103.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70640⟩⟩) (rawTerms := some (Proof.Events344.exact88108RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound88103.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound90087.bound, LeftBound88103.bound]
def bound : CoeffClass := .finite ⟨3456353380086899479155517117627148481331252, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90087.bound, LeftBound88103.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound90087.actual selector witness, LeftBound88103.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90092

namespace LeftBound90096
def owner : Owner := ⟨.program ⟨257⟩, ⟨70642⟩⟩
def transferEvent : Nat := 90096
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 90094 .coefficient, .predecessor 1 90095 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 90094 .coefficient)
      LeftBound90091.bound (LeftBound90091.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact90093RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90091.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90091.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 90095 .coefficient)
      LeftBound87889.bound (LeftBound87889.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact87896RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87889.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87889.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound90091.bound, LeftBound87889.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90091.bound, LeftBound87889.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound90091.actual selector witness, LeftBound87889.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90096

namespace LeftBound90097
def owner : Owner := ⟨.program ⟨257⟩, ⟨70642⟩⟩
def transferEvent : Nat := 90097
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 90093 .summary, .result 87896 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 90093 .summary)
      LeftBound90092.bound (LeftBound90092.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70641⟩⟩) (rawTerms := some (Proof.Events351.exact90093RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound90092.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 87896 .summary)
      LeftBound87891.bound (LeftBound87891.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28437⟩⟩) (rawTerms := some (Proof.Events343.exact87896RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound87891.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound90092.bound, LeftBound87891.bound]
def bound : CoeffClass := .finite ⟨3802007596962448506045899439491360353157172, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90092.bound, LeftBound87891.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound90092.actual selector witness, LeftBound87891.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90097

namespace LeftBound90101
def owner : Owner := ⟨.program ⟨257⟩, ⟨70643⟩⟩
def transferEvent : Nat := 90101
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 90099 .coefficient, .predecessor 1 90100 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 90099 .coefficient)
      LeftBound90096.bound (LeftBound90096.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact90098RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90096.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90096.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 90100 .coefficient)
      LeftBound87677.bound (LeftBound87677.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events342.exact87684RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87677.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87677.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound90096.bound, LeftBound87677.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90096.bound, LeftBound87677.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound90096.actual selector witness, LeftBound87677.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90101

namespace LeftBound90102
def owner : Owner := ⟨.program ⟨257⟩, ⟨70643⟩⟩
def transferEvent : Nat := 90102
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 90098 .summary, .result 87684 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 90098 .summary)
      LeftBound90097.bound (LeftBound90097.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70642⟩⟩) (rawTerms := some (Proof.Events351.exact90098RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound90097.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 87684 .summary)
      LeftBound87679.bound (LeftBound87679.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨31117⟩⟩) (rawTerms := some (Proof.Events342.exact87684RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound87679.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound90097.bound, LeftBound87679.bound]
def bound : CoeffClass := .finite ⟨4147668141949793872257454032897973461975092, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90097.bound, LeftBound87679.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound90097.actual selector witness, LeftBound87679.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90102

namespace LeftBound90106
def owner : Owner := ⟨.program ⟨257⟩, ⟨70644⟩⟩
def transferEvent : Nat := 90106
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 90104 .coefficient, .predecessor 1 90105 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 90104 .coefficient)
      LeftBound90101.bound (LeftBound90101.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact90103RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90101.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90101.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 90105 .coefficient)
      LeftBound87465.bound (LeftBound87465.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87472RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87465.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87465.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound90101.bound, LeftBound87465.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90101.bound, LeftBound87465.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound90101.actual selector witness, LeftBound87465.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90106

namespace LeftBound90107
def owner : Owner := ⟨.program ⟨257⟩, ⟨70644⟩⟩
def transferEvent : Nat := 90107
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 90103 .summary, .result 87472 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 90103 .summary)
      LeftBound90102.bound (LeftBound90102.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70643⟩⟩) (rawTerms := some (Proof.Events351.exact90103RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound90102.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 87472 .summary)
      LeftBound87467.bound (LeftBound87467.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36777⟩⟩) (rawTerms := some (Proof.Events341.exact87472RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound87467.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound90102.bound, LeftBound87467.bound]
def bound : CoeffClass := .finite ⟨4493332905678336798016456807332854062121012, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90102.bound, LeftBound87467.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound90102.actual selector witness, LeftBound87467.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90107

namespace LeftBound90111
def owner : Owner := ⟨.program ⟨257⟩, ⟨70645⟩⟩
def transferEvent : Nat := 90111
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 90109 .coefficient, .predecessor 1 90110 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 90109 .coefficient)
      LeftBound90106.bound (LeftBound90106.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact90108RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90106.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90106.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 90110 .coefficient)
      LeftBound87253.bound (LeftBound87253.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events340.exact87260RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87253.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87253.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound90106.bound, LeftBound87253.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90106.bound, LeftBound87253.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound90106.actual selector witness, LeftBound87253.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90111

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
