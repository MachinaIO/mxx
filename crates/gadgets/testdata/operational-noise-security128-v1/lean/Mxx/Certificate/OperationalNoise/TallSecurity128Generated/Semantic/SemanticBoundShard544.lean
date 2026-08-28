import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard503
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard507
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard510
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard511
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard514
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard518
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard521
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard525
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard529
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard532
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard543

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound84584
def owner : Owner := ⟨.program ⟨257⟩, ⟨34082⟩⟩
def transferEvent : Nat := 84584
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 84580 .summary, .result 83124 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 84580 .summary)
      LeftBound84579.bound (LeftBound84579.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨24062⟩⟩) (rawTerms := some (Proof.Events330.exact84580RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84579.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 83124 .summary)
      LeftBound83123.bound (LeftBound83123.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨34081⟩⟩) (rawTerms := some (Proof.Events324.exact83124RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound83123.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84579.bound, LeftBound83123.bound]
def bound : CoeffClass := .finite ⟨128755916426494733378385616044032, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84579.bound, LeftBound83123.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound84579.actual selector witness, LeftBound83123.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84584

namespace LeftBound84588
def owner : Owner := ⟨.program ⟨257⟩, ⟨53142⟩⟩
def transferEvent : Nat := 84588
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 84586 .coefficient, .predecessor 1 84587 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 84586 .coefficient)
      LeftBound84583.bound (LeftBound84583.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84585RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84583.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84583.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 84587 .coefficient)
      LeftBound82638.bound (LeftBound82638.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events322.exact82642RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82638.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82638.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84583.bound, LeftBound82638.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84583.bound, LeftBound82638.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound84583.actual selector witness, LeftBound82638.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84588

namespace LeftBound84589
def owner : Owner := ⟨.program ⟨257⟩, ⟨53142⟩⟩
def transferEvent : Nat := 84589
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 84585 .summary, .result 82642 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 84585 .summary)
      LeftBound84584.bound (LeftBound84584.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨34082⟩⟩) (rawTerms := some (Proof.Events330.exact84585RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84584.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 82642 .summary)
      LeftBound82641.bound (LeftBound82641.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨53141⟩⟩) (rawTerms := some (Proof.Events322.exact82642RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound82641.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84584.bound, LeftBound82641.bound]
def bound : CoeffClass := .finite ⟨160945509440761189776859800535040, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84584.bound, LeftBound82641.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound84584.actual selector witness, LeftBound82641.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84589

namespace LeftBound84593
def owner : Owner := ⟨.program ⟨257⟩, ⟨56122⟩⟩
def transferEvent : Nat := 84593
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 84591 .coefficient, .predecessor 1 84592 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 84591 .coefficient)
      LeftBound84588.bound (LeftBound84588.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84590RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84588.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84588.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 84592 .coefficient)
      LeftBound82156.bound (LeftBound82156.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events320.exact82160RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82156.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82156.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84588.bound, LeftBound82156.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84588.bound, LeftBound82156.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound84588.actual selector witness, LeftBound82156.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84593

namespace LeftBound84594
def owner : Owner := ⟨.program ⟨257⟩, ⟨56122⟩⟩
def transferEvent : Nat := 84594
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 84590 .summary, .result 82160 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 84590 .summary)
      LeftBound84589.bound (LeftBound84589.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨53142⟩⟩) (rawTerms := some (Proof.Events330.exact84590RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84589.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 82160 .summary)
      LeftBound82159.bound (LeftBound82159.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨56121⟩⟩) (rawTerms := some (Proof.Events320.exact82160RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound82159.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84589.bound, LeftBound82159.bound]
def bound : CoeffClass := .finite ⟨193135298905473333552574874779648, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84589.bound, LeftBound82159.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound84589.actual selector witness, LeftBound82159.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84594

namespace LeftBound84598
def owner : Owner := ⟨.program ⟨257⟩, ⟨59102⟩⟩
def transferEvent : Nat := 84598
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 84596 .coefficient, .predecessor 1 84597 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 84596 .coefficient)
      LeftBound84593.bound (LeftBound84593.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84595RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84593.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84593.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 84597 .coefficient)
      LeftBound81674.bound (LeftBound81674.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events319.exact81678RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81674.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81674.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84593.bound, LeftBound81674.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84593.bound, LeftBound81674.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound84593.actual selector witness, LeftBound81674.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84598

namespace LeftBound84599
def owner : Owner := ⟨.program ⟨257⟩, ⟨59102⟩⟩
def transferEvent : Nat := 84599
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 84595 .summary, .result 81678 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 84595 .summary)
      LeftBound84594.bound (LeftBound84594.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨56122⟩⟩) (rawTerms := some (Proof.Events330.exact84595RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84594.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 81678 .summary)
      LeftBound81677.bound (LeftBound81677.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨59101⟩⟩) (rawTerms := some (Proof.Events319.exact81678RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound81677.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84594.bound, LeftBound81677.bound]
def bound : CoeffClass := .finite ⟨225325481271076852082771728531456, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84594.bound, LeftBound81677.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound84594.actual selector witness, LeftBound81677.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84599

namespace LeftBound84603
def owner : Owner := ⟨.program ⟨257⟩, ⟨62082⟩⟩
def transferEvent : Nat := 84603
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 84601 .coefficient, .predecessor 1 84602 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 84601 .coefficient)
      LeftBound84598.bound (LeftBound84598.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84600RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84598.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84598.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 84602 .coefficient)
      LeftBound81192.bound (LeftBound81192.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events317.exact81196RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81192.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81192.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84598.bound, LeftBound81192.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84598.bound, LeftBound81192.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound84598.actual selector witness, LeftBound81192.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84603

namespace LeftBound84604
def owner : Owner := ⟨.program ⟨257⟩, ⟨62082⟩⟩
def transferEvent : Nat := 84604
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 84600 .summary, .result 81196 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 84600 .summary)
      LeftBound84599.bound (LeftBound84599.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨59102⟩⟩) (rawTerms := some (Proof.Events330.exact84600RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84599.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 81196 .summary)
      LeftBound81195.bound (LeftBound81195.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨62081⟩⟩) (rawTerms := some (Proof.Events317.exact81196RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound81195.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84599.bound, LeftBound81195.bound]
def bound : CoeffClass := .finite ⟨257515860087126057990209472036864, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84599.bound, LeftBound81195.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound84599.actual selector witness, LeftBound81195.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84604

namespace LeftBound84608
def owner : Owner := ⟨.program ⟨257⟩, ⟨65062⟩⟩
def transferEvent : Nat := 84608
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 84606 .coefficient, .predecessor 1 84607 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 84606 .coefficient)
      LeftBound84603.bound (LeftBound84603.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84605RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84603.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84603.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 84607 .coefficient)
      LeftBound80710.bound (LeftBound80710.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events315.exact80714RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80710.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80710.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84603.bound, LeftBound80710.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84603.bound, LeftBound80710.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound84603.actual selector witness, LeftBound80710.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84608

namespace LeftBound84609
def owner : Owner := ⟨.program ⟨257⟩, ⟨65062⟩⟩
def transferEvent : Nat := 84609
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 84605 .summary, .result 80714 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 84605 .summary)
      LeftBound84604.bound (LeftBound84604.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨62082⟩⟩) (rawTerms := some (Proof.Events330.exact84605RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84604.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 80714 .summary)
      LeftBound80713.bound (LeftBound80713.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨65061⟩⟩) (rawTerms := some (Proof.Events315.exact80714RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80713.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84604.bound, LeftBound80713.bound]
def bound : CoeffClass := .finite ⟨289706631804066638652128995049472, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84604.bound, LeftBound80713.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound84604.actual selector witness, LeftBound80713.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84609

namespace LeftBound84613
def owner : Owner := ⟨.program ⟨257⟩, ⟨70655⟩⟩
def transferEvent : Nat := 84613
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 84611 .coefficient, .predecessor 1 84612 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 84611 .coefficient)
      LeftBound84608.bound (LeftBound84608.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84610RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84608.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84608.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 84612 .coefficient)
      LeftBound80228.bound (LeftBound80228.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events313.exact80232RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80228.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80228.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84608.bound, LeftBound80228.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84608.bound, LeftBound80228.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound84608.actual selector witness, LeftBound80228.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84613

namespace LeftBound84614
def owner : Owner := ⟨.program ⟨257⟩, ⟨70655⟩⟩
def transferEvent : Nat := 84614
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 84610 .summary, .result 80232 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 84610 .summary)
      LeftBound84609.bound (LeftBound84609.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨65062⟩⟩) (rawTerms := some (Proof.Events330.exact84610RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84609.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 80232 .summary)
      LeftBound80231.bound (LeftBound80231.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70654⟩⟩) (rawTerms := some (Proof.Events313.exact80232RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80231.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84609.bound, LeftBound80231.bound]
def bound : CoeffClass := .finite ⟨321897992872344281445771187322880, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84609.bound, LeftBound80231.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound84609.actual selector witness, LeftBound80231.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84614

namespace LeftBound84618
def owner : Owner := ⟨.program ⟨257⟩, ⟨70656⟩⟩
def transferEvent : Nat := 84618
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 84616 .coefficient, .predecessor 1 84617 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 84616 .coefficient)
      LeftBound84613.bound (LeftBound84613.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84615RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84613.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84613.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 84617 .coefficient)
      LeftBound79746.bound (LeftBound79746.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events311.exact79750RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79746.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79746.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84613.bound, LeftBound79746.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84613.bound, LeftBound79746.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound84613.actual selector witness, LeftBound79746.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84618

namespace LeftBound84619
def owner : Owner := ⟨.program ⟨257⟩, ⟨70656⟩⟩
def transferEvent : Nat := 84619
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 84615 .summary, .result 79750 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 84615 .summary)
      LeftBound84614.bound (LeftBound84614.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70655⟩⟩) (rawTerms := some (Proof.Events330.exact84615RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84614.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 79750 .summary)
      LeftBound79749.bound (LeftBound79749.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28442⟩⟩) (rawTerms := some (Proof.Events311.exact79750RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79749.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84614.bound, LeftBound79749.bound]
def bound : CoeffClass := .finite ⟨354089550391067611616654269349888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84614.bound, LeftBound79749.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound84614.actual selector witness, LeftBound79749.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84619

namespace LeftBound84623
def owner : Owner := ⟨.program ⟨257⟩, ⟨70657⟩⟩
def transferEvent : Nat := 84623
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 84621 .coefficient, .predecessor 1 84622 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 84621 .coefficient)
      LeftBound84618.bound (LeftBound84618.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84620RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84618.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84618.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 84622 .coefficient)
      LeftBound79264.bound (LeftBound79264.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events309.exact79268RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79264.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79264.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84618.bound, LeftBound79264.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84618.bound, LeftBound79264.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound84618.actual selector witness, LeftBound79264.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84623

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
