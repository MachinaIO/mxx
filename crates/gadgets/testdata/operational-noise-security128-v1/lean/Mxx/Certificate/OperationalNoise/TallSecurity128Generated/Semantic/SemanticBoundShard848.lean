import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard818
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard822
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard826
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard829
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard833
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard837
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard840
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard844
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard847

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound128448
def owner : Owner := ⟨.program ⟨257⟩, ⟨20532⟩⟩
def transferEvent : Nat := 128448
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 128446 .coefficient, .predecessor 1 128447 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 128446 .coefficient)
      LeftBound128441.bound (LeftBound128441.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events501.exact128445RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound128441.bound, RecordedBoundRefines] <;> decide)
      (LeftBound128441.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 128447 .coefficient)
      LeftBound127959.bound (LeftBound127959.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events499.exact127963RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound127959.bound, RecordedBoundRefines] <;> decide)
      (LeftBound127959.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound128441.bound, LeftBound127959.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound128441.bound, LeftBound127959.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound128441.actual selector witness, LeftBound127959.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound128448

namespace LeftBound128449
def owner : Owner := ⟨.program ⟨257⟩, ⟨20532⟩⟩
def transferEvent : Nat := 128449
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 128445 .summary, .result 127963 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 128445 .summary)
      LeftBound128444.bound (LeftBound128444.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17652⟩⟩) (rawTerms := some (Proof.Events501.exact128445RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound128444.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 127963 .summary)
      LeftBound127962.bound (LeftBound127962.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20531⟩⟩) (rawTerms := some (Proof.Events499.exact127963RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound127962.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound128444.bound, LeftBound127962.bound]
def bound : CoeffClass := .finite ⟨64377712650190257467641695830016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound128444.bound, LeftBound127962.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound128444.actual selector witness, LeftBound127962.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound128449

namespace LeftBound128453
def owner : Owner := ⟨.program ⟨257⟩, ⟨23752⟩⟩
def transferEvent : Nat := 128453
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 128451 .coefficient, .predecessor 1 128452 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 128451 .coefficient)
      LeftBound128448.bound (LeftBound128448.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events501.exact128450RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound128448.bound, RecordedBoundRefines] <;> decide)
      (LeftBound128448.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 128452 .coefficient)
      LeftBound127477.bound (LeftBound127477.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events497.exact127481RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound127477.bound, RecordedBoundRefines] <;> decide)
      (LeftBound127477.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound128448.bound, LeftBound127477.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound128448.bound, LeftBound127477.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound128448.actual selector witness, LeftBound127477.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound128453

namespace LeftBound128454
def owner : Owner := ⟨.program ⟨257⟩, ⟨23752⟩⟩
def transferEvent : Nat := 128454
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 128450 .summary, .result 127481 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 128450 .summary)
      LeftBound128449.bound (LeftBound128449.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20532⟩⟩) (rawTerms := some (Proof.Events501.exact128450RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound128449.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 127481 .summary)
      LeftBound127480.bound (LeftBound127480.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23751⟩⟩) (rawTerms := some (Proof.Events497.exact127481RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound127480.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound128449.bound, LeftBound127480.bound]
def bound : CoeffClass := .finite ⟨96566716313119651734393211060224, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound128449.bound, LeftBound127480.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound128449.actual selector witness, LeftBound127480.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound128454

namespace LeftBound128458
def owner : Owner := ⟨.program ⟨257⟩, ⟨33772⟩⟩
def transferEvent : Nat := 128458
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 128456 .coefficient, .predecessor 1 128457 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 128456 .coefficient)
      LeftBound128453.bound (LeftBound128453.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events501.exact128455RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound128453.bound, RecordedBoundRefines] <;> decide)
      (LeftBound128453.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 128457 .coefficient)
      LeftBound126995.bound (LeftBound126995.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events496.exact126999RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound126995.bound, RecordedBoundRefines] <;> decide)
      (LeftBound126995.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound128453.bound, LeftBound126995.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound128453.bound, LeftBound126995.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound128453.actual selector witness, LeftBound126995.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound128458

namespace LeftBound128459
def owner : Owner := ⟨.program ⟨257⟩, ⟨33772⟩⟩
def transferEvent : Nat := 128459
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 128455 .summary, .result 126999 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 128455 .summary)
      LeftBound128454.bound (LeftBound128454.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23752⟩⟩) (rawTerms := some (Proof.Events501.exact128455RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound128454.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 126999 .summary)
      LeftBound126998.bound (LeftBound126998.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33771⟩⟩) (rawTerms := some (Proof.Events496.exact126999RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound126998.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound128454.bound, LeftBound126998.bound]
def bound : CoeffClass := .finite ⟨128755916426494733378385616044032, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound128454.bound, LeftBound126998.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound128454.actual selector witness, LeftBound126998.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound128459

namespace LeftBound128463
def owner : Owner := ⟨.program ⟨257⟩, ⟨52832⟩⟩
def transferEvent : Nat := 128463
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 128461 .coefficient, .predecessor 1 128462 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 128461 .coefficient)
      LeftBound128458.bound (LeftBound128458.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events501.exact128460RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound128458.bound, RecordedBoundRefines] <;> decide)
      (LeftBound128458.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 128462 .coefficient)
      LeftBound126513.bound (LeftBound126513.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events494.exact126517RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound126513.bound, RecordedBoundRefines] <;> decide)
      (LeftBound126513.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound128458.bound, LeftBound126513.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound128458.bound, LeftBound126513.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound128458.actual selector witness, LeftBound126513.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound128463

namespace LeftBound128464
def owner : Owner := ⟨.program ⟨257⟩, ⟨52832⟩⟩
def transferEvent : Nat := 128464
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 128460 .summary, .result 126517 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 128460 .summary)
      LeftBound128459.bound (LeftBound128459.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33772⟩⟩) (rawTerms := some (Proof.Events501.exact128460RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound128459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 126517 .summary)
      LeftBound126516.bound (LeftBound126516.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52831⟩⟩) (rawTerms := some (Proof.Events494.exact126517RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound126516.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound128459.bound, LeftBound126516.bound]
def bound : CoeffClass := .finite ⟨160945509440761189776859800535040, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound128459.bound, LeftBound126516.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound128459.actual selector witness, LeftBound126516.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound128464

namespace LeftBound128468
def owner : Owner := ⟨.program ⟨257⟩, ⟨55812⟩⟩
def transferEvent : Nat := 128468
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 128466 .coefficient, .predecessor 1 128467 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 128466 .coefficient)
      LeftBound128463.bound (LeftBound128463.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events501.exact128465RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound128463.bound, RecordedBoundRefines] <;> decide)
      (LeftBound128463.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 128467 .coefficient)
      LeftBound126031.bound (LeftBound126031.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events492.exact126035RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound126031.bound, RecordedBoundRefines] <;> decide)
      (LeftBound126031.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound128463.bound, LeftBound126031.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound128463.bound, LeftBound126031.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound128463.actual selector witness, LeftBound126031.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound128468

namespace LeftBound128469
def owner : Owner := ⟨.program ⟨257⟩, ⟨55812⟩⟩
def transferEvent : Nat := 128469
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 128465 .summary, .result 126035 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 128465 .summary)
      LeftBound128464.bound (LeftBound128464.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52832⟩⟩) (rawTerms := some (Proof.Events501.exact128465RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound128464.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 126035 .summary)
      LeftBound126034.bound (LeftBound126034.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55811⟩⟩) (rawTerms := some (Proof.Events492.exact126035RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound126034.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound128464.bound, LeftBound126034.bound]
def bound : CoeffClass := .finite ⟨193135298905473333552574874779648, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound128464.bound, LeftBound126034.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound128464.actual selector witness, LeftBound126034.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound128469

namespace LeftBound128473
def owner : Owner := ⟨.program ⟨257⟩, ⟨58792⟩⟩
def transferEvent : Nat := 128473
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 128471 .coefficient, .predecessor 1 128472 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 128471 .coefficient)
      LeftBound128468.bound (LeftBound128468.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events501.exact128470RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound128468.bound, RecordedBoundRefines] <;> decide)
      (LeftBound128468.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 128472 .coefficient)
      LeftBound125549.bound (LeftBound125549.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events490.exact125553RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound125549.bound, RecordedBoundRefines] <;> decide)
      (LeftBound125549.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound128468.bound, LeftBound125549.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound128468.bound, LeftBound125549.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound128468.actual selector witness, LeftBound125549.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound128473

namespace LeftBound128474
def owner : Owner := ⟨.program ⟨257⟩, ⟨58792⟩⟩
def transferEvent : Nat := 128474
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 128470 .summary, .result 125553 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 128470 .summary)
      LeftBound128469.bound (LeftBound128469.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55812⟩⟩) (rawTerms := some (Proof.Events501.exact128470RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound128469.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 125553 .summary)
      LeftBound125552.bound (LeftBound125552.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58791⟩⟩) (rawTerms := some (Proof.Events490.exact125553RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound125552.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound128469.bound, LeftBound125552.bound]
def bound : CoeffClass := .finite ⟨225325481271076852082771728531456, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound128469.bound, LeftBound125552.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound128469.actual selector witness, LeftBound125552.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound128474

namespace LeftBound128478
def owner : Owner := ⟨.program ⟨257⟩, ⟨61772⟩⟩
def transferEvent : Nat := 128478
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 128476 .coefficient, .predecessor 1 128477 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 128476 .coefficient)
      LeftBound128473.bound (LeftBound128473.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events501.exact128475RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound128473.bound, RecordedBoundRefines] <;> decide)
      (LeftBound128473.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 128477 .coefficient)
      LeftBound125067.bound (LeftBound125067.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events488.exact125071RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound125067.bound, RecordedBoundRefines] <;> decide)
      (LeftBound125067.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound128473.bound, LeftBound125067.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound128473.bound, LeftBound125067.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound128473.actual selector witness, LeftBound125067.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound128478

namespace LeftBound128479
def owner : Owner := ⟨.program ⟨257⟩, ⟨61772⟩⟩
def transferEvent : Nat := 128479
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 128475 .summary, .result 125071 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 128475 .summary)
      LeftBound128474.bound (LeftBound128474.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58792⟩⟩) (rawTerms := some (Proof.Events501.exact128475RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound128474.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 125071 .summary)
      LeftBound125070.bound (LeftBound125070.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61771⟩⟩) (rawTerms := some (Proof.Events488.exact125071RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound125070.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound128474.bound, LeftBound125070.bound]
def bound : CoeffClass := .finite ⟨257515860087126057990209472036864, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound128474.bound, LeftBound125070.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound128474.actual selector witness, LeftBound125070.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound128479

namespace LeftBound128483
def owner : Owner := ⟨.program ⟨257⟩, ⟨64752⟩⟩
def transferEvent : Nat := 128483
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 128481 .coefficient, .predecessor 1 128482 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 128481 .coefficient)
      LeftBound128478.bound (LeftBound128478.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events501.exact128480RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound128478.bound, RecordedBoundRefines] <;> decide)
      (LeftBound128478.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 128482 .coefficient)
      LeftBound124585.bound (LeftBound124585.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events486.exact124589RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound124585.bound, RecordedBoundRefines] <;> decide)
      (LeftBound124585.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound128478.bound, LeftBound124585.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound128478.bound, LeftBound124585.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound128478.actual selector witness, LeftBound124585.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound128483

namespace LeftBound128484
def owner : Owner := ⟨.program ⟨257⟩, ⟨64752⟩⟩
def transferEvent : Nat := 128484
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 128480 .summary, .result 124589 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 128480 .summary)
      LeftBound128479.bound (LeftBound128479.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61772⟩⟩) (rawTerms := some (Proof.Events501.exact128480RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound128479.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 124589 .summary)
      LeftBound124588.bound (LeftBound124588.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64751⟩⟩) (rawTerms := some (Proof.Events486.exact124589RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound124588.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound128479.bound, LeftBound124588.bound]
def bound : CoeffClass := .finite ⟨289706631804066638652128995049472, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound128479.bound, LeftBound124588.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound128479.actual selector witness, LeftBound124588.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound128484

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
