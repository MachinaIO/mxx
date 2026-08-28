import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard064
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard066
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard109

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound22531
def owner : Owner := ⟨.program ⟨257⟩, ⟨59937⟩⟩
def transferEvent : Nat := 22531
def frameStart : Nat := 22443
def rule : BoundRule := .product (.predecessor 0 22529 .coefficient) (.predecessor 1 22530 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 22529 .coefficient)
      LeftAuthority22504.bound (LeftAuthority22504.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events087.exact22505RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22504.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22504.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 22530 .coefficient)
      LeftAuthority22527.bound (LeftAuthority22527.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22528RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22527.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22527.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority22504.bound LeftAuthority22527.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority22504.bound, LeftAuthority22527.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority22504.actual selector witness) * (LeftAuthority22527.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22531

namespace LeftBound22539
def owner : Owner := ⟨.program ⟨257⟩, ⟨59938⟩⟩
def transferEvent : Nat := 22539
def frameStart : Nat := 22443
def rule : BoundRule := .sum [.predecessor 0 22537 .coefficient, .predecessor 1 22538 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 22537 .coefficient)
      LeftAuthority22535.bound (LeftAuthority22535.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22536RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22535.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22535.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 22538 .coefficient)
      LeftBound22531.bound (LeftBound22531.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22533RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22531.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22531.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority22535.bound, LeftBound22531.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority22535.bound, LeftBound22531.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority22535.actual selector witness, LeftBound22531.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound22539

namespace LeftBound22543
def owner : Owner := ⟨.program ⟨257⟩, ⟨61627⟩⟩
def transferEvent : Nat := 22543
def frameStart : Nat := 22443
def rule : BoundRule := .sum [.predecessor 0 22541 .coefficient, .predecessor 1 22542 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 22541 .coefficient)
      LeftBound22539.bound (LeftBound22539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22539.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22539.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 22542 .coefficient)
      LeftBound22520.bound (LeftBound22520.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events087.exact22525RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22520.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22520.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound22539.bound, LeftBound22520.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22539.bound, LeftBound22520.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound22539.actual selector witness, LeftBound22520.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound22543

namespace LeftBound22556
def owner : Owner := ⟨.program ⟨257⟩, ⟨61625⟩⟩
def transferEvent : Nat := 22556
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 22554 .coefficient, .predecessor 1 22555 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 22554 .coefficient)
      LeftBound22385.bound (LeftBound22385.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22553RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22385.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 22555 .coefficient)
      LeftBound22368.bound (LeftBound22368.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events087.exact22375RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22368.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22368.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound22385.bound, LeftBound22368.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22385.bound, LeftBound22368.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound22385.actual selector witness, LeftBound22368.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound22556

namespace LeftBound22559
def owner : Owner := ⟨.program ⟨257⟩, ⟨61625⟩⟩
def transferEvent : Nat := 22559
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 22553 .summary, .result 22375 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 22553 .summary)
      LeftBound22387.bound (LeftBound22387.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨60525⟩⟩) (rawTerms := some (Proof.Events088.exact22553RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound22387.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 22375 .summary)
      LeftBound22370.bound (LeftBound22370.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61624⟩⟩) (rawTerms := some (Proof.Events087.exact22375RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound22370.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound22387.bound, LeftBound22370.bound]
def bound : CoeffClass := .finite ⟨32190378816049205907437743505408, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22387.bound, LeftBound22370.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound22387.actual selector witness, LeftBound22370.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound22559

namespace LeftBound22582
def owner : Owner := ⟨.program ⟨257⟩, ⟨99⟩⟩
def transferEvent : Nat := 22582
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 22581 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 22581 .coefficient)
      LeftAuthority17048.bound (LeftAuthority17048.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events066.exact17049RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17048.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17048.derived selector witness)

def rawBound : CoeffClass := LeftAuthority17048.bound
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17048.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority17048.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound22582

namespace LeftBound22586
def owner : Owner := ⟨.program ⟨257⟩, ⟨24907⟩⟩
def transferEvent : Nat := 22586
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 22584 .coefficient) (.predecessor 1 22585 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 22584 .coefficient)
      LeftAuthority303.bound (LeftAuthority303.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events001.exact304RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority303.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority303.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 22585 .coefficient)
      LeftBound17055.bound (LeftBound17055.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events066.exact17057RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17055.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17055.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority303.bound LeftBound17055.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority303.bound, LeftBound17055.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority303.actual selector witness) * (LeftBound17055.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound22586

namespace LeftBound22590
def owner : Owner := ⟨.program ⟨257⟩, ⟨7273⟩⟩
def transferEvent : Nat := 22590
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 22589 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 22589 .coefficient)
      LeftAuthority15892.bound (LeftAuthority15892.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events062.exact15893RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15892.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15892.derived selector witness)

def rawBound : CoeffClass := LeftAuthority15892.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15892.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority15892.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound22590

namespace LeftBound22594
def owner : Owner := ⟨.program ⟨257⟩, ⟨7591⟩⟩
def transferEvent : Nat := 22594
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 22592 .coefficient) (.predecessor 1 22593 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 22592 .coefficient)
      LeftBound16921.bound (LeftBound16921.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events066.exact16922RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16921.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16921.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 22593 .coefficient)
      LeftBound22590.bound (LeftBound22590.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22591RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22590.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22590.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound16921.bound LeftBound22590.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound16921.bound, LeftBound22590.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound16921.actual selector witness) * (LeftBound22590.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22594

namespace LeftBound22599
def owner : Owner := ⟨.program ⟨257⟩, ⟨24908⟩⟩
def transferEvent : Nat := 22599
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 22597 .coefficient, .predecessor 1 22598 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 22597 .coefficient)
      LeftBound22594.bound (LeftBound22594.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22596RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22594.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22594.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 22598 .coefficient)
      LeftBound22586.bound (LeftBound22586.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22588RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22586.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22586.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound22594.bound, LeftBound22586.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22594.bound, LeftBound22586.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound22594.actual selector witness, LeftBound22586.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound22599

namespace LeftBound22603
def owner : Owner := ⟨.program ⟨257⟩, ⟨24909⟩⟩
def transferEvent : Nat := 22603
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 22601 .coefficient, .predecessor 1 22602 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 22601 .coefficient)
      LeftBound22599.bound (LeftBound22599.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22600RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22599.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22599.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 22602 .coefficient)
      LeftBound22582.bound (LeftBound22582.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22583RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22582.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22582.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound22599.bound, LeftBound22582.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22599.bound, LeftBound22582.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound22599.actual selector witness, LeftBound22582.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound22603

namespace LeftBound22604
def owner : Owner := ⟨.program ⟨257⟩, ⟨24909⟩⟩
def transferEvent : Nat := 22604
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨99⟩⟩]⟩ [⟨.result 22583 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 22583 .coefficient)
      LeftBound22582.bound (LeftBound22582.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨99⟩⟩) (rawTerms := some (Proof.Events088.exact22583RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22582.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22582.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound22582.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22582.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound22582.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound22604

namespace LeftBound22609
def owner : Owner := ⟨.program ⟨257⟩, ⟨56274⟩⟩
def transferEvent : Nat := 22609
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 22607 .coefficient) (.predecessor 1 22608 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 22607 .coefficient)
      LeftBound22603.bound (LeftBound22603.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22606RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22603.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22603.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 22608 .coefficient)
      LeftAuthority306.bound (LeftAuthority306.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events001.exact307RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority306.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority306.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound22603.bound LeftAuthority306.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22603.bound, LeftAuthority306.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound22603.actual selector witness) * (LeftAuthority306.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22609

namespace LeftBound22610
def owner : Owner := ⟨.program ⟨257⟩, ⟨56274⟩⟩
def transferEvent : Nat := 22610
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨56271⟩⟩], []⟩ [⟨.result 307 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 307 .coefficient)
      LeftAuthority306.bound (LeftAuthority306.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨56271⟩⟩) (rawTerms := some (Proof.Events001.exact307RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority306.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority306.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority306.bound []
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority306.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority306.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound22610

namespace LeftBound22611
def owner : Owner := ⟨.program ⟨257⟩, ⟨56274⟩⟩
def transferEvent : Nat := 22611
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 22606 .summary) (.transfer 22610) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 22606 .summary)
      LeftBound22604.bound (LeftBound22604.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨24909⟩⟩) (rawTerms := some (Proof.Events088.exact22606RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound22604.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 22610)
      LeftBound22610.bound (LeftBound22610.actual selector witness) := by
  exact .transfer (LeftBound22610.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound22604.bound LeftBound22610.bound
def bound : CoeffClass := .finite ⟨13631488, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22604.bound, LeftBound22610.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound22604.actual selector witness) * (LeftBound22610.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22611

namespace LeftBound22620
def owner : Owner := ⟨.program ⟨257⟩, ⟨9533⟩⟩
def transferEvent : Nat := 22620
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 22618 .coefficient) (.value (.predecessor 1 22619 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 22618 .coefficient)
      LeftAuthority22616.bound (LeftAuthority22616.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22617RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22616.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22616.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 22619 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority22616.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority22616.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority22616.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound22620

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
