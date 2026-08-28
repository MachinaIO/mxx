import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard175
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard215

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound37559
def owner : Owner := ⟨.program ⟨257⟩, ⟨9534⟩⟩
def transferEvent : Nat := 37559
def frameStart : Nat := 37477
def rule : BoundRule := .product (.predecessor 0 37557 .coefficient) (.predecessor 1 37558 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 37557 .coefficient)
      LeftBound37555.bound (LeftBound37555.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events146.exact37556RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37555.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37555.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 37558 .coefficient)
      LeftBound37552.bound (LeftBound37552.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events146.exact37553RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37552.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37552.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound37555.bound LeftBound37552.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37555.bound, LeftBound37552.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound37555.actual selector witness) * (LeftBound37552.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound37559

namespace LeftBound37564
def owner : Owner := ⟨.program ⟨257⟩, ⟨58285⟩⟩
def transferEvent : Nat := 37564
def frameStart : Nat := 37477
def rule : BoundRule := .sum [.predecessor 0 37562 .coefficient, .predecessor 1 37563 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 37562 .coefficient)
      LeftBound37559.bound (LeftBound37559.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events146.exact37561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37559.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 37563 .coefficient)
      LeftBound37536.bound (LeftBound37536.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events146.exact37538RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37536.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37536.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound37559.bound, LeftBound37536.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37559.bound, LeftBound37536.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound37559.actual selector witness, LeftBound37536.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound37564

namespace LeftBound37568
def owner : Owner := ⟨.program ⟨257⟩, ⟨58581⟩⟩
def transferEvent : Nat := 37568
def frameStart : Nat := 37477
def rule : BoundRule := .product (.predecessor 0 37566 .coefficient) (.predecessor 1 37567 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 37566 .coefficient)
      LeftBound37564.bound (LeftBound37564.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events146.exact37565RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37564.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37564.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 37567 .coefficient)
      LeftAuthority37521.bound (LeftAuthority37521.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events146.exact37522RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37521.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37521.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound37564.bound LeftAuthority37521.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37564.bound, LeftAuthority37521.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound37564.actual selector witness) * (LeftAuthority37521.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound37568

namespace LeftBound37579
def owner : Owner := ⟨.program ⟨257⟩, ⟨56922⟩⟩
def transferEvent : Nat := 37579
def frameStart : Nat := 37477
def rule : BoundRule := .product (.predecessor 0 37577 .coefficient) (.predecessor 1 37578 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 37577 .coefficient)
      LeftAuthority37532.bound (LeftAuthority37532.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events146.exact37533RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37532.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37532.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 37578 .coefficient)
      LeftAuthority37575.bound (LeftAuthority37575.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events146.exact37576RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37575.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37575.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority37532.bound LeftAuthority37575.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority37532.bound, LeftAuthority37575.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority37532.actual selector witness) * (LeftAuthority37575.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound37579

namespace LeftBound37587
def owner : Owner := ⟨.program ⟨257⟩, ⟨56923⟩⟩
def transferEvent : Nat := 37587
def frameStart : Nat := 37477
def rule : BoundRule := .sum [.predecessor 0 37585 .coefficient, .predecessor 1 37586 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 37585 .coefficient)
      LeftAuthority37583.bound (LeftAuthority37583.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events146.exact37584RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37583.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37583.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 37586 .coefficient)
      LeftBound37579.bound (LeftBound37579.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events146.exact37581RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37579.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37579.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority37583.bound, LeftBound37579.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority37583.bound, LeftBound37579.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority37583.actual selector witness, LeftBound37579.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound37587

namespace LeftBound37591
def owner : Owner := ⟨.program ⟨257⟩, ⟨58582⟩⟩
def transferEvent : Nat := 37591
def frameStart : Nat := 37477
def rule : BoundRule := .sum [.predecessor 0 37589 .coefficient, .predecessor 1 37590 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 37589 .coefficient)
      LeftBound37587.bound (LeftBound37587.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events146.exact37588RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37587.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37587.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 37590 .coefficient)
      LeftBound37568.bound (LeftBound37568.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events146.exact37573RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37568.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37568.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound37587.bound, LeftBound37568.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37587.bound, LeftBound37568.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound37587.actual selector witness, LeftBound37568.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound37591

namespace LeftBound37604
def owner : Owner := ⟨.program ⟨257⟩, ⟨58580⟩⟩
def transferEvent : Nat := 37604
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 37602 .coefficient, .predecessor 1 37603 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 37602 .coefficient)
      LeftBound37425.bound (LeftBound37425.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events146.exact37601RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37425.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37425.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 37603 .coefficient)
      LeftBound37408.bound (LeftBound37408.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events146.exact37415RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37408.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37408.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound37425.bound, LeftBound37408.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37425.bound, LeftBound37408.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound37425.actual selector witness, LeftBound37408.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound37604

namespace LeftBound37607
def owner : Owner := ⟨.program ⟨257⟩, ⟨58580⟩⟩
def transferEvent : Nat := 37607
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 37601 .summary, .result 37415 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 37601 .summary)
      LeftBound37427.bound (LeftBound37427.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨57502⟩⟩) (rawTerms := some (Proof.Events146.exact37601RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound37427.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 37415 .summary)
      LeftBound37410.bound (LeftBound37410.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58579⟩⟩) (rawTerms := some (Proof.Events146.exact37415RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound37410.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound37427.bound, LeftBound37410.bound]
def bound : CoeffClass := .finite ⟨2997944351807545540608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37427.bound, LeftBound37410.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound37427.actual selector witness, LeftBound37410.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound37607

namespace LeftBound37611
def owner : Owner := ⟨.program ⟨257⟩, ⟨59193⟩⟩
def transferEvent : Nat := 37611
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 37609 .coefficient) (.predecessor 1 37610 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 37609 .coefficient)
      LeftBound37604.bound (LeftBound37604.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events146.exact37608RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37604.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37604.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 37610 .coefficient)
      LeftAuthority37330.bound (LeftAuthority37330.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events145.exact37331RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37330.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37330.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound37604.bound LeftAuthority37330.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37604.bound, LeftAuthority37330.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound37604.actual selector witness) * (LeftAuthority37330.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound37611

namespace LeftBound37612
def owner : Owner := ⟨.program ⟨257⟩, ⟨59193⟩⟩
def transferEvent : Nat := 37612
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨59191⟩⟩]⟩ [⟨.result 37331 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 37331 .coefficient)
      LeftAuthority37330.bound (LeftAuthority37330.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨59191⟩⟩) (rawTerms := some (Proof.Events145.exact37331RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37330.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37330.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority37330.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority37330.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority37330.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound37612

namespace LeftBound37613
def owner : Owner := ⟨.program ⟨257⟩, ⟨59193⟩⟩
def transferEvent : Nat := 37613
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 37608 .summary) (.transfer 37612) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 37608 .summary)
      LeftBound37607.bound (LeftBound37607.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58580⟩⟩) (rawTerms := some (Proof.Events146.exact37608RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound37607.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 37612)
      LeftBound37612.bound (LeftBound37612.actual selector witness) := by
  exact .transfer (LeftBound37612.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound37607.bound LeftBound37612.bound
def bound : CoeffClass := .finite ⟨32190182365603316457354999889920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37607.bound, LeftBound37612.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound37607.actual selector witness) * (LeftBound37612.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound37613

namespace LeftBound37624
def owner : Owner := ⟨.program ⟨257⟩, ⟨57898⟩⟩
def transferEvent : Nat := 37624
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 37622 .coefficient) (.value (.predecessor 1 37623 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 37622 .coefficient)
      LeftAuthority37620.bound (LeftAuthority37620.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events146.exact37621RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37620.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37620.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 37623 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority37620.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority37620.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority37620.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound37624

namespace LeftBound37628
def owner : Owner := ⟨.program ⟨257⟩, ⟨57899⟩⟩
def transferEvent : Nat := 37628
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 37626 .coefficient) (.predecessor 1 37627 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 37626 .coefficient)
      LeftBound32117.bound (LeftBound32117.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events125.exact32120RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32117.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32117.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 37627 .coefficient)
      LeftBound37624.bound (LeftBound37624.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events146.exact37625RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37624.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37624.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound32117.bound LeftBound37624.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32117.bound, LeftBound37624.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound32117.actual selector witness) * (LeftBound37624.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound37628

namespace LeftBound37629
def owner : Owner := ⟨.program ⟨257⟩, ⟨57899⟩⟩
def transferEvent : Nat := 37629
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨57896⟩⟩]⟩ [⟨.result 37621 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 37621 .coefficient)
      LeftAuthority37620.bound (LeftAuthority37620.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨57896⟩⟩) (rawTerms := some (Proof.Events146.exact37621RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37620.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37620.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority37620.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority37620.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority37620.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound37629

namespace LeftBound37630
def owner : Owner := ⟨.program ⟨257⟩, ⟨57899⟩⟩
def transferEvent : Nat := 37630
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 32120 .summary) (.transfer 37629) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 32120 .summary)
      LeftBound32118.bound (LeftBound32118.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨11643⟩⟩) (rawTerms := some (Proof.Events125.exact32120RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound32118.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 37629)
      LeftBound37629.bound (LeftBound37629.actual selector witness) := by
  exact .transfer (LeftBound37629.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound32118.bound LeftBound37629.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32118.bound, LeftBound37629.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound32118.actual selector witness) * (LeftBound37629.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound37630

namespace LeftBound37725
def owner : Owner := ⟨.program ⟨257⟩, ⟨56921⟩⟩
def transferEvent : Nat := 37725
def frameStart : Nat := 37686
def rule : BoundRule := .identity (.predecessor 0 37724 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 37724 .coefficient)
      LeftAuthority37722.bound (LeftAuthority37722.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events147.exact37723RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37722.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37722.derived selector witness)

def rawBound : CoeffClass := LeftAuthority37722.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority37722.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority37722.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound37725

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
