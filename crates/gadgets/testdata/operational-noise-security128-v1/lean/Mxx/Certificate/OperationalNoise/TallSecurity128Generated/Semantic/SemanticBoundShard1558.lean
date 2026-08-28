import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1528
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1532
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1536
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1539
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1543
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1547
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1550
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1554
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1557

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound230819
def owner : Owner := ⟨.program ⟨257⟩, ⟨17736⟩⟩
def transferEvent : Nat := 230819
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 230813 .summary, .result 230635 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 230813 .summary)
      LeftBound230647.bound (LeftBound230647.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨16579⟩⟩) (rawTerms := some (Proof.Events901.exact230813RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound230647.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 230635 .summary)
      LeftBound230630.bound (LeftBound230630.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17735⟩⟩) (rawTerms := some (Proof.Events900.exact230635RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound230630.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound230647.bound, LeftBound230630.bound]
def bound : CoeffClass := .finite ⟨32188807212483706889510625476608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound230647.bound, LeftBound230630.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound230647.actual selector witness, LeftBound230630.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound230819

namespace LeftBound230823
def owner : Owner := ⟨.program ⟨257⟩, ⟨20625⟩⟩
def transferEvent : Nat := 230823
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 230821 .coefficient, .predecessor 1 230822 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 230821 .coefficient)
      LeftBound230816.bound (LeftBound230816.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events901.exact230820RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound230816.bound, RecordedBoundRefines] <;> decide)
      (LeftBound230816.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 230822 .coefficient)
      LeftBound230334.bound (LeftBound230334.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events899.exact230338RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound230334.bound, RecordedBoundRefines] <;> decide)
      (LeftBound230334.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound230816.bound, LeftBound230334.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound230816.bound, LeftBound230334.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound230816.actual selector witness, LeftBound230334.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound230823

namespace LeftBound230824
def owner : Owner := ⟨.program ⟨257⟩, ⟨20625⟩⟩
def transferEvent : Nat := 230824
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 230820 .summary, .result 230338 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 230820 .summary)
      LeftBound230819.bound (LeftBound230819.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17736⟩⟩) (rawTerms := some (Proof.Events901.exact230820RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound230819.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 230338 .summary)
      LeftBound230337.bound (LeftBound230337.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20624⟩⟩) (rawTerms := some (Proof.Events899.exact230338RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound230337.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound230819.bound, LeftBound230337.bound]
def bound : CoeffClass := .finite ⟨64377712650190257467641695830016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound230819.bound, LeftBound230337.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound230819.actual selector witness, LeftBound230337.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound230824

namespace LeftBound230828
def owner : Owner := ⟨.program ⟨257⟩, ⟨23845⟩⟩
def transferEvent : Nat := 230828
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 230826 .coefficient, .predecessor 1 230827 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 230826 .coefficient)
      LeftBound230823.bound (LeftBound230823.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events901.exact230825RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound230823.bound, RecordedBoundRefines] <;> decide)
      (LeftBound230823.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 230827 .coefficient)
      LeftBound229852.bound (LeftBound229852.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events897.exact229856RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound229852.bound, RecordedBoundRefines] <;> decide)
      (LeftBound229852.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound230823.bound, LeftBound229852.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound230823.bound, LeftBound229852.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound230823.actual selector witness, LeftBound229852.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound230828

namespace LeftBound230829
def owner : Owner := ⟨.program ⟨257⟩, ⟨23845⟩⟩
def transferEvent : Nat := 230829
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 230825 .summary, .result 229856 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 230825 .summary)
      LeftBound230824.bound (LeftBound230824.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20625⟩⟩) (rawTerms := some (Proof.Events901.exact230825RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound230824.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 229856 .summary)
      LeftBound229855.bound (LeftBound229855.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23844⟩⟩) (rawTerms := some (Proof.Events897.exact229856RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound229855.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound230824.bound, LeftBound229855.bound]
def bound : CoeffClass := .finite ⟨96566716313119651734393211060224, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound230824.bound, LeftBound229855.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound230824.actual selector witness, LeftBound229855.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound230829

namespace LeftBound230833
def owner : Owner := ⟨.program ⟨257⟩, ⟨33865⟩⟩
def transferEvent : Nat := 230833
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 230831 .coefficient, .predecessor 1 230832 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 230831 .coefficient)
      LeftBound230828.bound (LeftBound230828.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events901.exact230830RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound230828.bound, RecordedBoundRefines] <;> decide)
      (LeftBound230828.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 230832 .coefficient)
      LeftBound229370.bound (LeftBound229370.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events895.exact229374RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound229370.bound, RecordedBoundRefines] <;> decide)
      (LeftBound229370.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound230828.bound, LeftBound229370.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound230828.bound, LeftBound229370.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound230828.actual selector witness, LeftBound229370.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound230833

namespace LeftBound230834
def owner : Owner := ⟨.program ⟨257⟩, ⟨33865⟩⟩
def transferEvent : Nat := 230834
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 230830 .summary, .result 229374 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 230830 .summary)
      LeftBound230829.bound (LeftBound230829.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23845⟩⟩) (rawTerms := some (Proof.Events901.exact230830RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound230829.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 229374 .summary)
      LeftBound229373.bound (LeftBound229373.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33864⟩⟩) (rawTerms := some (Proof.Events895.exact229374RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound229373.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound230829.bound, LeftBound229373.bound]
def bound : CoeffClass := .finite ⟨128755916426494733378385616044032, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound230829.bound, LeftBound229373.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound230829.actual selector witness, LeftBound229373.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound230834

namespace LeftBound230838
def owner : Owner := ⟨.program ⟨257⟩, ⟨52925⟩⟩
def transferEvent : Nat := 230838
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 230836 .coefficient, .predecessor 1 230837 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 230836 .coefficient)
      LeftBound230833.bound (LeftBound230833.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events901.exact230835RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound230833.bound, RecordedBoundRefines] <;> decide)
      (LeftBound230833.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 230837 .coefficient)
      LeftBound228888.bound (LeftBound228888.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events894.exact228892RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound228888.bound, RecordedBoundRefines] <;> decide)
      (LeftBound228888.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound230833.bound, LeftBound228888.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound230833.bound, LeftBound228888.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound230833.actual selector witness, LeftBound228888.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound230838

namespace LeftBound230839
def owner : Owner := ⟨.program ⟨257⟩, ⟨52925⟩⟩
def transferEvent : Nat := 230839
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 230835 .summary, .result 228892 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 230835 .summary)
      LeftBound230834.bound (LeftBound230834.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33865⟩⟩) (rawTerms := some (Proof.Events901.exact230835RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound230834.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 228892 .summary)
      LeftBound228891.bound (LeftBound228891.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52924⟩⟩) (rawTerms := some (Proof.Events894.exact228892RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound228891.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound230834.bound, LeftBound228891.bound]
def bound : CoeffClass := .finite ⟨160945509440761189776859800535040, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound230834.bound, LeftBound228891.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound230834.actual selector witness, LeftBound228891.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound230839

namespace LeftBound230843
def owner : Owner := ⟨.program ⟨257⟩, ⟨55905⟩⟩
def transferEvent : Nat := 230843
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 230841 .coefficient, .predecessor 1 230842 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 230841 .coefficient)
      LeftBound230838.bound (LeftBound230838.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events901.exact230840RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound230838.bound, RecordedBoundRefines] <;> decide)
      (LeftBound230838.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 230842 .coefficient)
      LeftBound228406.bound (LeftBound228406.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events892.exact228410RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound228406.bound, RecordedBoundRefines] <;> decide)
      (LeftBound228406.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound230838.bound, LeftBound228406.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound230838.bound, LeftBound228406.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound230838.actual selector witness, LeftBound228406.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound230843

namespace LeftBound230844
def owner : Owner := ⟨.program ⟨257⟩, ⟨55905⟩⟩
def transferEvent : Nat := 230844
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 230840 .summary, .result 228410 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 230840 .summary)
      LeftBound230839.bound (LeftBound230839.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52925⟩⟩) (rawTerms := some (Proof.Events901.exact230840RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound230839.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 228410 .summary)
      LeftBound228409.bound (LeftBound228409.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55904⟩⟩) (rawTerms := some (Proof.Events892.exact228410RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound228409.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound230839.bound, LeftBound228409.bound]
def bound : CoeffClass := .finite ⟨193135298905473333552574874779648, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound230839.bound, LeftBound228409.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound230839.actual selector witness, LeftBound228409.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound230844

namespace LeftBound230848
def owner : Owner := ⟨.program ⟨257⟩, ⟨58885⟩⟩
def transferEvent : Nat := 230848
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 230846 .coefficient, .predecessor 1 230847 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 230846 .coefficient)
      LeftBound230843.bound (LeftBound230843.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events901.exact230845RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound230843.bound, RecordedBoundRefines] <;> decide)
      (LeftBound230843.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 230847 .coefficient)
      LeftBound227924.bound (LeftBound227924.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events890.exact227928RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound227924.bound, RecordedBoundRefines] <;> decide)
      (LeftBound227924.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound230843.bound, LeftBound227924.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound230843.bound, LeftBound227924.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound230843.actual selector witness, LeftBound227924.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound230848

namespace LeftBound230849
def owner : Owner := ⟨.program ⟨257⟩, ⟨58885⟩⟩
def transferEvent : Nat := 230849
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 230845 .summary, .result 227928 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 230845 .summary)
      LeftBound230844.bound (LeftBound230844.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55905⟩⟩) (rawTerms := some (Proof.Events901.exact230845RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound230844.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 227928 .summary)
      LeftBound227927.bound (LeftBound227927.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58884⟩⟩) (rawTerms := some (Proof.Events890.exact227928RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound227927.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound230844.bound, LeftBound227927.bound]
def bound : CoeffClass := .finite ⟨225325481271076852082771728531456, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound230844.bound, LeftBound227927.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound230844.actual selector witness, LeftBound227927.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound230849

namespace LeftBound230853
def owner : Owner := ⟨.program ⟨257⟩, ⟨61865⟩⟩
def transferEvent : Nat := 230853
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 230851 .coefficient, .predecessor 1 230852 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 230851 .coefficient)
      LeftBound230848.bound (LeftBound230848.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events901.exact230850RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound230848.bound, RecordedBoundRefines] <;> decide)
      (LeftBound230848.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 230852 .coefficient)
      LeftBound227442.bound (LeftBound227442.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events888.exact227446RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound227442.bound, RecordedBoundRefines] <;> decide)
      (LeftBound227442.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound230848.bound, LeftBound227442.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound230848.bound, LeftBound227442.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound230848.actual selector witness, LeftBound227442.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound230853

namespace LeftBound230854
def owner : Owner := ⟨.program ⟨257⟩, ⟨61865⟩⟩
def transferEvent : Nat := 230854
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 230850 .summary, .result 227446 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 230850 .summary)
      LeftBound230849.bound (LeftBound230849.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58885⟩⟩) (rawTerms := some (Proof.Events901.exact230850RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound230849.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 227446 .summary)
      LeftBound227445.bound (LeftBound227445.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61864⟩⟩) (rawTerms := some (Proof.Events888.exact227446RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound227445.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound230849.bound, LeftBound227445.bound]
def bound : CoeffClass := .finite ⟨257515860087126057990209472036864, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound230849.bound, LeftBound227445.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound230849.actual selector witness, LeftBound227445.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound230854

namespace LeftBound230858
def owner : Owner := ⟨.program ⟨257⟩, ⟨64845⟩⟩
def transferEvent : Nat := 230858
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 230856 .coefficient, .predecessor 1 230857 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 230856 .coefficient)
      LeftBound230853.bound (LeftBound230853.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events901.exact230855RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound230853.bound, RecordedBoundRefines] <;> decide)
      (LeftBound230853.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 230857 .coefficient)
      LeftBound226960.bound (LeftBound226960.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events886.exact226964RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound226960.bound, RecordedBoundRefines] <;> decide)
      (LeftBound226960.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound230853.bound, LeftBound226960.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound230853.bound, LeftBound226960.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound230853.actual selector witness, LeftBound226960.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound230858

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
