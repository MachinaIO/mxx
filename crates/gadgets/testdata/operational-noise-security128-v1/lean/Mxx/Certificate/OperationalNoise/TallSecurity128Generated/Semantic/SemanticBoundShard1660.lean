import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1615
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1619
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1623
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1626
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1630
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1634
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1637
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1641
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1659

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound245468
def owner : Owner := ⟨.program ⟨257⟩, ⟨55874⟩⟩
def transferEvent : Nat := 245468
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 245466 .coefficient, .predecessor 1 245467 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 245466 .coefficient)
      LeftBound245463.bound (LeftBound245463.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events958.exact245465RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound245463.bound, RecordedBoundRefines] <;> decide)
      (LeftBound245463.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 245467 .coefficient)
      LeftBound243031.bound (LeftBound243031.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events949.exact243035RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound243031.bound, RecordedBoundRefines] <;> decide)
      (LeftBound243031.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound245463.bound, LeftBound243031.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound245463.bound, LeftBound243031.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound245463.actual selector witness, LeftBound243031.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound245468

namespace LeftBound245469
def owner : Owner := ⟨.program ⟨257⟩, ⟨55874⟩⟩
def transferEvent : Nat := 245469
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 245465 .summary, .result 243035 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 245465 .summary)
      LeftBound245464.bound (LeftBound245464.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52894⟩⟩) (rawTerms := some (Proof.Events958.exact245465RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound245464.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 243035 .summary)
      LeftBound243034.bound (LeftBound243034.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55873⟩⟩) (rawTerms := some (Proof.Events949.exact243035RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound243034.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound245464.bound, LeftBound243034.bound]
def bound : CoeffClass := .finite ⟨193135298905473333552574874779648, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound245464.bound, LeftBound243034.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound245464.actual selector witness, LeftBound243034.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound245469

namespace LeftBound245473
def owner : Owner := ⟨.program ⟨257⟩, ⟨58854⟩⟩
def transferEvent : Nat := 245473
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 245471 .coefficient, .predecessor 1 245472 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 245471 .coefficient)
      LeftBound245468.bound (LeftBound245468.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events958.exact245470RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound245468.bound, RecordedBoundRefines] <;> decide)
      (LeftBound245468.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 245472 .coefficient)
      LeftBound242549.bound (LeftBound242549.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events947.exact242553RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound242549.bound, RecordedBoundRefines] <;> decide)
      (LeftBound242549.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound245468.bound, LeftBound242549.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound245468.bound, LeftBound242549.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound245468.actual selector witness, LeftBound242549.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound245473

namespace LeftBound245474
def owner : Owner := ⟨.program ⟨257⟩, ⟨58854⟩⟩
def transferEvent : Nat := 245474
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 245470 .summary, .result 242553 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 245470 .summary)
      LeftBound245469.bound (LeftBound245469.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55874⟩⟩) (rawTerms := some (Proof.Events958.exact245470RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound245469.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 242553 .summary)
      LeftBound242552.bound (LeftBound242552.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58853⟩⟩) (rawTerms := some (Proof.Events947.exact242553RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound242552.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound245469.bound, LeftBound242552.bound]
def bound : CoeffClass := .finite ⟨225325481271076852082771728531456, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound245469.bound, LeftBound242552.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound245469.actual selector witness, LeftBound242552.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound245474

namespace LeftBound245478
def owner : Owner := ⟨.program ⟨257⟩, ⟨61834⟩⟩
def transferEvent : Nat := 245478
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 245476 .coefficient, .predecessor 1 245477 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 245476 .coefficient)
      LeftBound245473.bound (LeftBound245473.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events958.exact245475RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound245473.bound, RecordedBoundRefines] <;> decide)
      (LeftBound245473.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 245477 .coefficient)
      LeftBound242067.bound (LeftBound242067.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events945.exact242071RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound242067.bound, RecordedBoundRefines] <;> decide)
      (LeftBound242067.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound245473.bound, LeftBound242067.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound245473.bound, LeftBound242067.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound245473.actual selector witness, LeftBound242067.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound245478

namespace LeftBound245479
def owner : Owner := ⟨.program ⟨257⟩, ⟨61834⟩⟩
def transferEvent : Nat := 245479
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 245475 .summary, .result 242071 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 245475 .summary)
      LeftBound245474.bound (LeftBound245474.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58854⟩⟩) (rawTerms := some (Proof.Events958.exact245475RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound245474.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 242071 .summary)
      LeftBound242070.bound (LeftBound242070.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61833⟩⟩) (rawTerms := some (Proof.Events945.exact242071RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound242070.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound245474.bound, LeftBound242070.bound]
def bound : CoeffClass := .finite ⟨257515860087126057990209472036864, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound245474.bound, LeftBound242070.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound245474.actual selector witness, LeftBound242070.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound245479

namespace LeftBound245483
def owner : Owner := ⟨.program ⟨257⟩, ⟨64814⟩⟩
def transferEvent : Nat := 245483
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 245481 .coefficient, .predecessor 1 245482 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 245481 .coefficient)
      LeftBound245478.bound (LeftBound245478.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events958.exact245480RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound245478.bound, RecordedBoundRefines] <;> decide)
      (LeftBound245478.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 245482 .coefficient)
      LeftBound241585.bound (LeftBound241585.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events943.exact241589RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound241585.bound, RecordedBoundRefines] <;> decide)
      (LeftBound241585.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound245478.bound, LeftBound241585.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound245478.bound, LeftBound241585.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound245478.actual selector witness, LeftBound241585.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound245483

namespace LeftBound245484
def owner : Owner := ⟨.program ⟨257⟩, ⟨64814⟩⟩
def transferEvent : Nat := 245484
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 245480 .summary, .result 241589 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 245480 .summary)
      LeftBound245479.bound (LeftBound245479.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61834⟩⟩) (rawTerms := some (Proof.Events958.exact245480RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound245479.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 241589 .summary)
      LeftBound241588.bound (LeftBound241588.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64813⟩⟩) (rawTerms := some (Proof.Events943.exact241589RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound241588.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound245479.bound, LeftBound241588.bound]
def bound : CoeffClass := .finite ⟨289706631804066638652128995049472, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound245479.bound, LeftBound241588.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound245479.actual selector witness, LeftBound241588.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound245484

namespace LeftBound245488
def owner : Owner := ⟨.program ⟨257⟩, ⟨70023⟩⟩
def transferEvent : Nat := 245488
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 245486 .coefficient, .predecessor 1 245487 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 245486 .coefficient)
      LeftBound245483.bound (LeftBound245483.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events958.exact245485RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound245483.bound, RecordedBoundRefines] <;> decide)
      (LeftBound245483.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 245487 .coefficient)
      LeftBound241103.bound (LeftBound241103.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events941.exact241107RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound241103.bound, RecordedBoundRefines] <;> decide)
      (LeftBound241103.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound245483.bound, LeftBound241103.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound245483.bound, LeftBound241103.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound245483.actual selector witness, LeftBound241103.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound245488

namespace LeftBound245489
def owner : Owner := ⟨.program ⟨257⟩, ⟨70023⟩⟩
def transferEvent : Nat := 245489
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 245485 .summary, .result 241107 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 245485 .summary)
      LeftBound245484.bound (LeftBound245484.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64814⟩⟩) (rawTerms := some (Proof.Events958.exact245485RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound245484.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 241107 .summary)
      LeftBound241106.bound (LeftBound241106.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70022⟩⟩) (rawTerms := some (Proof.Events941.exact241107RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound241106.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound245484.bound, LeftBound241106.bound]
def bound : CoeffClass := .finite ⟨321897992872344281445771187322880, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound245484.bound, LeftBound241106.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound245484.actual selector witness, LeftBound241106.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound245489

namespace LeftBound245493
def owner : Owner := ⟨.program ⟨257⟩, ⟨70024⟩⟩
def transferEvent : Nat := 245493
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 245491 .coefficient, .predecessor 1 245492 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 245491 .coefficient)
      LeftBound245488.bound (LeftBound245488.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events958.exact245490RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound245488.bound, RecordedBoundRefines] <;> decide)
      (LeftBound245488.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 245492 .coefficient)
      LeftBound240621.bound (LeftBound240621.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events939.exact240625RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound240621.bound, RecordedBoundRefines] <;> decide)
      (LeftBound240621.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound245488.bound, LeftBound240621.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound245488.bound, LeftBound240621.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound245488.actual selector witness, LeftBound240621.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound245493

namespace LeftBound245494
def owner : Owner := ⟨.program ⟨257⟩, ⟨70024⟩⟩
def transferEvent : Nat := 245494
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 245490 .summary, .result 240625 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 245490 .summary)
      LeftBound245489.bound (LeftBound245489.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70023⟩⟩) (rawTerms := some (Proof.Events958.exact245490RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound245489.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 240625 .summary)
      LeftBound240624.bound (LeftBound240624.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28242⟩⟩) (rawTerms := some (Proof.Events939.exact240625RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound240624.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound245489.bound, LeftBound240624.bound]
def bound : CoeffClass := .finite ⟨354089550391067611616654269349888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound245489.bound, LeftBound240624.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound245489.actual selector witness, LeftBound240624.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound245494

namespace LeftBound245498
def owner : Owner := ⟨.program ⟨257⟩, ⟨70025⟩⟩
def transferEvent : Nat := 245498
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 245496 .coefficient, .predecessor 1 245497 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 245496 .coefficient)
      LeftBound245493.bound (LeftBound245493.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events958.exact245495RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound245493.bound, RecordedBoundRefines] <;> decide)
      (LeftBound245493.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 245497 .coefficient)
      LeftBound240139.bound (LeftBound240139.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events938.exact240143RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound240139.bound, RecordedBoundRefines] <;> decide)
      (LeftBound240139.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound245493.bound, LeftBound240139.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound245493.bound, LeftBound240139.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound245493.actual selector witness, LeftBound240139.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound245498

namespace LeftBound245499
def owner : Owner := ⟨.program ⟨257⟩, ⟨70025⟩⟩
def transferEvent : Nat := 245499
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 245495 .summary, .result 240143 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 245495 .summary)
      LeftBound245494.bound (LeftBound245494.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70024⟩⟩) (rawTerms := some (Proof.Events958.exact245495RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound245494.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 240143 .summary)
      LeftBound240142.bound (LeftBound240142.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨30922⟩⟩) (rawTerms := some (Proof.Events938.exact240143RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound240142.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound245494.bound, LeftBound240142.bound]
def bound : CoeffClass := .finite ⟨386281697261128003919260020637696, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound245494.bound, LeftBound240142.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound245494.actual selector witness, LeftBound240142.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound245499

namespace LeftBound245503
def owner : Owner := ⟨.program ⟨257⟩, ⟨70026⟩⟩
def transferEvent : Nat := 245503
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 245501 .coefficient, .predecessor 1 245502 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 245501 .coefficient)
      LeftBound245498.bound (LeftBound245498.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events958.exact245500RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound245498.bound, RecordedBoundRefines] <;> decide)
      (LeftBound245498.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 245502 .coefficient)
      LeftBound239657.bound (LeftBound239657.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events936.exact239661RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound239657.bound, RecordedBoundRefines] <;> decide)
      (LeftBound239657.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound245498.bound, LeftBound239657.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound245498.bound, LeftBound239657.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound245498.actual selector witness, LeftBound239657.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound245503

namespace LeftBound245504
def owner : Owner := ⟨.program ⟨257⟩, ⟨70026⟩⟩
def transferEvent : Nat := 245504
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 245500 .summary, .result 239661 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 245500 .summary)
      LeftBound245499.bound (LeftBound245499.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70025⟩⟩) (rawTerms := some (Proof.Events958.exact245500RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound245499.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 239661 .summary)
      LeftBound239660.bound (LeftBound239660.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36582⟩⟩) (rawTerms := some (Proof.Events936.exact239661RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound239660.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound245499.bound, LeftBound239660.bound]
def bound : CoeffClass := .finite ⟨418474237032079770976347551432704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound245499.bound, LeftBound239660.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound245499.actual selector witness, LeftBound239660.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound245504

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
