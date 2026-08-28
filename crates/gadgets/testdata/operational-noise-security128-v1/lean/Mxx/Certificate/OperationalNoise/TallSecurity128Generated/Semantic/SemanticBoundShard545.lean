import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard481
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard482
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard485
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard489
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard492
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard496
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard500
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard503
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard544

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound84624
def owner : Owner := ⟨.program ⟨257⟩, ⟨70657⟩⟩
def transferEvent : Nat := 84624
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 84620 .summary, .result 79268 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 84620 .summary)
      LeftBound84619.bound (LeftBound84619.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70656⟩⟩) (rawTerms := some (Proof.Events330.exact84620RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84619.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 79268 .summary)
      LeftBound79267.bound (LeftBound79267.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨31122⟩⟩) (rawTerms := some (Proof.Events309.exact79268RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79267.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84619.bound, LeftBound79267.bound]
def bound : CoeffClass := .finite ⟨386281697261128003919260020637696, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84619.bound, LeftBound79267.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound84619.actual selector witness, LeftBound79267.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84624

namespace LeftBound84628
def owner : Owner := ⟨.program ⟨257⟩, ⟨70658⟩⟩
def transferEvent : Nat := 84628
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 84626 .coefficient, .predecessor 1 84627 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 84626 .coefficient)
      LeftBound84623.bound (LeftBound84623.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84625RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84623.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84623.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 84627 .coefficient)
      LeftBound78782.bound (LeftBound78782.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events307.exact78786RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78782.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78782.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84623.bound, LeftBound78782.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84623.bound, LeftBound78782.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound84623.actual selector witness, LeftBound78782.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84628

namespace LeftBound84629
def owner : Owner := ⟨.program ⟨257⟩, ⟨70658⟩⟩
def transferEvent : Nat := 84629
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 84625 .summary, .result 78786 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 84625 .summary)
      LeftBound84624.bound (LeftBound84624.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70657⟩⟩) (rawTerms := some (Proof.Events330.exact84625RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84624.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 78786 .summary)
      LeftBound78785.bound (LeftBound78785.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36782⟩⟩) (rawTerms := some (Proof.Events307.exact78786RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound78785.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84624.bound, LeftBound78785.bound]
def bound : CoeffClass := .finite ⟨418474237032079770976347551432704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84624.bound, LeftBound78785.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound84624.actual selector witness, LeftBound78785.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84629

namespace LeftBound84633
def owner : Owner := ⟨.program ⟨257⟩, ⟨70659⟩⟩
def transferEvent : Nat := 84633
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 84631 .coefficient, .predecessor 1 84632 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 84631 .coefficient)
      LeftBound84628.bound (LeftBound84628.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84630RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84628.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84628.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 84632 .coefficient)
      LeftBound78300.bound (LeftBound78300.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78304RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78300.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78300.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84628.bound, LeftBound78300.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84628.bound, LeftBound78300.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound84628.actual selector witness, LeftBound78300.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84633

namespace LeftBound84634
def owner : Owner := ⟨.program ⟨257⟩, ⟨70659⟩⟩
def transferEvent : Nat := 84634
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 84630 .summary, .result 78304 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 84630 .summary)
      LeftBound84629.bound (LeftBound84629.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70658⟩⟩) (rawTerms := some (Proof.Events330.exact84630RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84629.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 78304 .summary)
      LeftBound78303.bound (LeftBound78303.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39462⟩⟩) (rawTerms := some (Proof.Events305.exact78304RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound78303.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84629.bound, LeftBound78303.bound]
def bound : CoeffClass := .finite ⟨450666973253477225410675971981312, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84629.bound, LeftBound78303.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound84629.actual selector witness, LeftBound78303.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84634

namespace LeftBound84638
def owner : Owner := ⟨.program ⟨257⟩, ⟨70660⟩⟩
def transferEvent : Nat := 84638
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 84636 .coefficient, .predecessor 1 84637 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 84636 .coefficient)
      LeftBound84633.bound (LeftBound84633.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84635RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84633.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84633.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 84637 .coefficient)
      LeftBound77818.bound (LeftBound77818.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events303.exact77822RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77818.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77818.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84633.bound, LeftBound77818.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84633.bound, LeftBound77818.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound84633.actual selector witness, LeftBound77818.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84638

namespace LeftBound84639
def owner : Owner := ⟨.program ⟨257⟩, ⟨70660⟩⟩
def transferEvent : Nat := 84639
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 84635 .summary, .result 77822 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 84635 .summary)
      LeftBound84634.bound (LeftBound84634.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70659⟩⟩) (rawTerms := some (Proof.Events330.exact84635RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84634.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 77822 .summary)
      LeftBound77821.bound (LeftBound77821.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨42142⟩⟩) (rawTerms := some (Proof.Events303.exact77822RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound77821.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84634.bound, LeftBound77821.bound]
def bound : CoeffClass := .finite ⟨482860102375766054599486172037120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84634.bound, LeftBound77821.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound84634.actual selector witness, LeftBound77821.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84639

namespace LeftBound84643
def owner : Owner := ⟨.program ⟨257⟩, ⟨70661⟩⟩
def transferEvent : Nat := 84643
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 84641 .coefficient, .predecessor 1 84642 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 84641 .coefficient)
      LeftBound84638.bound (LeftBound84638.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84640RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84638.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84638.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 84642 .coefficient)
      LeftBound77336.bound (LeftBound77336.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events302.exact77340RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77336.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77336.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84638.bound, LeftBound77336.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84638.bound, LeftBound77336.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound84638.actual selector witness, LeftBound77336.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84643

namespace LeftBound84644
def owner : Owner := ⟨.program ⟨257⟩, ⟨70661⟩⟩
def transferEvent : Nat := 84644
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 84640 .summary, .result 77340 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 84640 .summary)
      LeftBound84639.bound (LeftBound84639.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70660⟩⟩) (rawTerms := some (Proof.Events330.exact84640RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84639.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 77340 .summary)
      LeftBound77339.bound (LeftBound77339.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44822⟩⟩) (rawTerms := some (Proof.Events302.exact77340RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound77339.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84639.bound, LeftBound77339.bound]
def bound : CoeffClass := .finite ⟨515053820849391945920019041353728, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84639.bound, LeftBound77339.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound84639.actual selector witness, LeftBound77339.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84644

namespace LeftBound84648
def owner : Owner := ⟨.program ⟨257⟩, ⟨70662⟩⟩
def transferEvent : Nat := 84648
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 84646 .coefficient, .predecessor 1 84647 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 84646 .coefficient)
      LeftBound84643.bound (LeftBound84643.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84645RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84643.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84643.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 84647 .coefficient)
      LeftBound76854.bound (LeftBound76854.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events300.exact76858RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76854.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76854.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84643.bound, LeftBound76854.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84643.bound, LeftBound76854.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound84643.actual selector witness, LeftBound76854.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84648

namespace LeftBound84649
def owner : Owner := ⟨.program ⟨257⟩, ⟨70662⟩⟩
def transferEvent : Nat := 84649
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 84645 .summary, .result 76858 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 84645 .summary)
      LeftBound84644.bound (LeftBound84644.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70661⟩⟩) (rawTerms := some (Proof.Events330.exact84645RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84644.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 76858 .summary)
      LeftBound76857.bound (LeftBound76857.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨47502⟩⟩) (rawTerms := some (Proof.Events300.exact76858RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound76857.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84644.bound, LeftBound76857.bound]
def bound : CoeffClass := .finite ⟨547248128674354899372274579931136, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84644.bound, LeftBound76857.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound84644.actual selector witness, LeftBound76857.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84649

namespace LeftBound84653
def owner : Owner := ⟨.program ⟨257⟩, ⟨70663⟩⟩
def transferEvent : Nat := 84653
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 84651 .coefficient, .predecessor 1 84652 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 84651 .coefficient)
      LeftBound84648.bound (LeftBound84648.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84650RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84648.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84648.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 84652 .coefficient)
      LeftBound76372.bound (LeftBound76372.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events298.exact76376RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76372.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76372.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84648.bound, LeftBound76372.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84648.bound, LeftBound76372.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound84648.actual selector witness, LeftBound76372.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84653

namespace LeftBound84654
def owner : Owner := ⟨.program ⟨257⟩, ⟨70663⟩⟩
def transferEvent : Nat := 84654
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 84650 .summary, .result 76376 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 84650 .summary)
      LeftBound84649.bound (LeftBound84649.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70662⟩⟩) (rawTerms := some (Proof.Events330.exact84650RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84649.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 76376 .summary)
      LeftBound76375.bound (LeftBound76375.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨50182⟩⟩) (rawTerms := some (Proof.Events298.exact76376RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound76375.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84649.bound, LeftBound76375.bound]
def bound : CoeffClass := .finite ⟨579442632949763540201771008262144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84649.bound, LeftBound76375.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound84649.actual selector witness, LeftBound76375.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84654

namespace LeftBound84658
def owner : Owner := ⟨.program ⟨257⟩, ⟨71439⟩⟩
def transferEvent : Nat := 84658
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 84656 .coefficient) (.predecessor 1 84657 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 84656 .coefficient)
      LeftBound84653.bound (LeftBound84653.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84655RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84653.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84653.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 84657 .coefficient)
      LeftAuthority75877.bound (LeftAuthority75877.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events296.exact75878RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75877.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75877.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound84653.bound LeftAuthority75877.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84653.bound, LeftAuthority75877.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound84653.actual selector witness) * (LeftAuthority75877.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84658

namespace LeftBound84659
def owner : Owner := ⟨.program ⟨257⟩, ⟨71439⟩⟩
def transferEvent : Nat := 84659
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩ [⟨.result 75878 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 75878 .coefficient)
      LeftAuthority75877.bound (LeftAuthority75877.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨71437⟩⟩) (rawTerms := some (Proof.Events296.exact75878RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75877.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75877.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority75877.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority75877.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority75877.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound84659

namespace LeftBound84660
def owner : Owner := ⟨.program ⟨257⟩, ⟨71439⟩⟩
def transferEvent : Nat := 84660
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 84655 .summary) (.transfer 84659) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 84655 .summary)
      LeftBound84654.bound (LeftBound84654.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70663⟩⟩) (rawTerms := some (Proof.Events330.exact84655RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84654.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 84659)
      LeftBound84659.bound (LeftBound84659.actual selector witness) := by
  exact .transfer (LeftBound84659.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound84654.bound LeftBound84659.bound
def bound : CoeffClass := .finite ⟨6221717896068416040249469304417135687106560, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84654.bound, LeftBound84659.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound84654.actual selector witness) * (LeftBound84659.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84660

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
