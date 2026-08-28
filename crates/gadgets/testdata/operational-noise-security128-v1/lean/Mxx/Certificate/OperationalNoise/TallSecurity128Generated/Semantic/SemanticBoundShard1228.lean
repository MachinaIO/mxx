import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard110
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard111
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1185
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1188
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1227

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound183531
def owner : Owner := ⟨.program ⟨257⟩, ⟨61986⟩⟩
def transferEvent : Nat := 183531
def frameStart : Nat := 183454
def rule : BoundRule := .product (.predecessor 0 183529 .coefficient) (.predecessor 1 183530 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 183529 .coefficient)
      LeftBound183527.bound (LeftBound183527.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events716.exact183528RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound183527.bound, RecordedBoundRefines] <;> decide)
      (LeftBound183527.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 183530 .coefficient)
      LeftAuthority183504.bound (LeftAuthority183504.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events716.exact183505RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority183504.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority183504.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound183527.bound LeftAuthority183504.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound183527.bound, LeftAuthority183504.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound183527.actual selector witness) * (LeftAuthority183504.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound183531

namespace LeftBound183542
def owner : Owner := ⟨.program ⟨257⟩, ⟨60160⟩⟩
def transferEvent : Nat := 183542
def frameStart : Nat := 183454
def rule : BoundRule := .product (.predecessor 0 183540 .coefficient) (.predecessor 1 183541 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 183540 .coefficient)
      LeftAuthority183515.bound (LeftAuthority183515.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events716.exact183516RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority183515.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority183515.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 183541 .coefficient)
      LeftAuthority183538.bound (LeftAuthority183538.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events716.exact183539RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority183538.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority183538.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority183515.bound LeftAuthority183538.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority183515.bound, LeftAuthority183538.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority183515.actual selector witness) * (LeftAuthority183538.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound183542

namespace LeftBound183550
def owner : Owner := ⟨.program ⟨257⟩, ⟨60161⟩⟩
def transferEvent : Nat := 183550
def frameStart : Nat := 183454
def rule : BoundRule := .sum [.predecessor 0 183548 .coefficient, .predecessor 1 183549 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 183548 .coefficient)
      LeftAuthority183546.bound (LeftAuthority183546.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events716.exact183547RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority183546.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority183546.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 183549 .coefficient)
      LeftBound183542.bound (LeftBound183542.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events716.exact183544RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound183542.bound, RecordedBoundRefines] <;> decide)
      (LeftBound183542.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority183546.bound, LeftBound183542.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority183546.bound, LeftBound183542.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority183546.actual selector witness, LeftBound183542.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound183550

namespace LeftBound183554
def owner : Owner := ⟨.program ⟨257⟩, ⟨61990⟩⟩
def transferEvent : Nat := 183554
def frameStart : Nat := 183454
def rule : BoundRule := .sum [.predecessor 0 183552 .coefficient, .predecessor 1 183553 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 183552 .coefficient)
      LeftBound183550.bound (LeftBound183550.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events716.exact183551RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound183550.bound, RecordedBoundRefines] <;> decide)
      (LeftBound183550.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 183553 .coefficient)
      LeftBound183531.bound (LeftBound183531.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events716.exact183536RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound183531.bound, RecordedBoundRefines] <;> decide)
      (LeftBound183531.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound183550.bound, LeftBound183531.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound183550.bound, LeftBound183531.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound183550.actual selector witness, LeftBound183531.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound183554

namespace LeftBound183567
def owner : Owner := ⟨.program ⟨257⟩, ⟨61988⟩⟩
def transferEvent : Nat := 183567
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 183565 .coefficient, .predecessor 1 183566 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 183565 .coefficient)
      LeftBound183396.bound (LeftBound183396.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events717.exact183564RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound183396.bound, RecordedBoundRefines] <;> decide)
      (LeftBound183396.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 183566 .coefficient)
      LeftBound183379.bound (LeftBound183379.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events716.exact183386RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound183379.bound, RecordedBoundRefines] <;> decide)
      (LeftBound183379.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound183396.bound, LeftBound183379.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound183396.bound, LeftBound183379.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound183396.actual selector witness, LeftBound183379.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound183567

namespace LeftBound183570
def owner : Owner := ⟨.program ⟨257⟩, ⟨61988⟩⟩
def transferEvent : Nat := 183570
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 183564 .summary, .result 183386 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 183564 .summary)
      LeftBound183398.bound (LeftBound183398.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨60759⟩⟩) (rawTerms := some (Proof.Events717.exact183564RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound183398.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 183386 .summary)
      LeftBound183381.bound (LeftBound183381.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61987⟩⟩) (rawTerms := some (Proof.Events716.exact183386RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound183381.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound183398.bound, LeftBound183381.bound]
def bound : CoeffClass := .finite ⟨32190378816049205907437743505408, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound183398.bound, LeftBound183381.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound183398.actual selector witness, LeftBound183381.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound183570

namespace LeftBound183594
def owner : Owner := ⟨.program ⟨257⟩, ⟨25047⟩⟩
def transferEvent : Nat := 183594
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 183592 .coefficient) (.predecessor 1 183593 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 183592 .coefficient)
      LeftAuthority8574.bound (LeftAuthority8574.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events033.exact8575RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8574.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8574.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 183593 .coefficient)
      LeftBound178276.bound (LeftBound178276.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events696.exact178278RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound178276.bound, RecordedBoundRefines] <;> decide)
      (LeftBound178276.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority8574.bound LeftBound178276.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority8574.bound, LeftBound178276.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority8574.actual selector witness) * (LeftBound178276.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound183594

namespace LeftBound183599
def owner : Owner := ⟨.program ⟨257⟩, ⟨8921⟩⟩
def transferEvent : Nat := 183599
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 183597 .coefficient) (.predecessor 1 183598 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 183597 .coefficient)
      LeftBound178147.bound (LeftBound178147.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events695.exact178148RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound178147.bound, RecordedBoundRefines] <;> decide)
      (LeftBound178147.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 183598 .coefficient)
      LeftBound22590.bound (LeftBound22590.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22591RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22590.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22590.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound178147.bound LeftBound22590.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound178147.bound, LeftBound22590.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound178147.actual selector witness) * (LeftBound22590.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound183599

namespace LeftBound183604
def owner : Owner := ⟨.program ⟨257⟩, ⟨25048⟩⟩
def transferEvent : Nat := 183604
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 183602 .coefficient, .predecessor 1 183603 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 183602 .coefficient)
      LeftBound183599.bound (LeftBound183599.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events717.exact183601RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound183599.bound, RecordedBoundRefines] <;> decide)
      (LeftBound183599.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 183603 .coefficient)
      LeftBound183594.bound (LeftBound183594.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events717.exact183596RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound183594.bound, RecordedBoundRefines] <;> decide)
      (LeftBound183594.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound183599.bound, LeftBound183594.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound183599.bound, LeftBound183594.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound183599.actual selector witness, LeftBound183594.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound183604

namespace LeftBound183608
def owner : Owner := ⟨.program ⟨257⟩, ⟨25049⟩⟩
def transferEvent : Nat := 183608
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 183606 .coefficient, .predecessor 1 183607 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 183606 .coefficient)
      LeftBound183604.bound (LeftBound183604.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events717.exact183605RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound183604.bound, RecordedBoundRefines] <;> decide)
      (LeftBound183604.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 183607 .coefficient)
      LeftBound22582.bound (LeftBound22582.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22583RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22582.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22582.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound183604.bound, LeftBound22582.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound183604.bound, LeftBound22582.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound183604.actual selector witness, LeftBound22582.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound183608

namespace LeftBound183609
def owner : Owner := ⟨.program ⟨257⟩, ⟨25049⟩⟩
def transferEvent : Nat := 183609
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
end LeftBound183609

namespace LeftBound183614
def owner : Owner := ⟨.program ⟨257⟩, ⟨56589⟩⟩
def transferEvent : Nat := 183614
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 183612 .coefficient) (.predecessor 1 183613 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 183612 .coefficient)
      LeftBound183608.bound (LeftBound183608.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events717.exact183611RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound183608.bound, RecordedBoundRefines] <;> decide)
      (LeftBound183608.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 183613 .coefficient)
      LeftAuthority8577.bound (LeftAuthority8577.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events033.exact8578RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8577.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8577.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound183608.bound LeftAuthority8577.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound183608.bound, LeftAuthority8577.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound183608.actual selector witness) * (LeftAuthority8577.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound183614

namespace LeftBound183615
def owner : Owner := ⟨.program ⟨257⟩, ⟨56589⟩⟩
def transferEvent : Nat := 183615
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨56586⟩⟩], []⟩ [⟨.result 8578 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 8578 .coefficient)
      LeftAuthority8577.bound (LeftAuthority8577.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨56586⟩⟩) (rawTerms := some (Proof.Events033.exact8578RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8577.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8577.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority8577.bound []
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority8577.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority8577.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound183615

namespace LeftBound183616
def owner : Owner := ⟨.program ⟨257⟩, ⟨56589⟩⟩
def transferEvent : Nat := 183616
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 183611 .summary) (.transfer 183615) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 183611 .summary)
      LeftBound183609.bound (LeftBound183609.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨25049⟩⟩) (rawTerms := some (Proof.Events717.exact183611RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound183609.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 183615)
      LeftBound183615.bound (LeftBound183615.actual selector witness) := by
  exact .transfer (LeftBound183615.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound183609.bound LeftBound183615.bound
def bound : CoeffClass := .finite ⟨13631488, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound183609.bound, LeftBound183615.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound183609.actual selector witness) * (LeftBound183615.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound183616

namespace LeftBound183622
def owner : Owner := ⟨.program ⟨257⟩, ⟨56590⟩⟩
def transferEvent : Nat := 183622
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 183620 .coefficient) (.predecessor 1 183621 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 183620 .coefficient)
      LeftAuthority8577.bound (LeftAuthority8577.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events033.exact8578RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8577.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8577.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 183621 .coefficient)
      LeftBound178276.bound (LeftBound178276.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events696.exact178278RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound178276.bound, RecordedBoundRefines] <;> decide)
      (LeftBound178276.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority8577.bound LeftBound178276.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority8577.bound, LeftBound178276.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority8577.actual selector witness) * (LeftBound178276.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound183622

namespace LeftBound183627
def owner : Owner := ⟨.program ⟨257⟩, ⟨8938⟩⟩
def transferEvent : Nat := 183627
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 183625 .coefficient) (.predecessor 1 183626 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 183625 .coefficient)
      LeftBound178147.bound (LeftBound178147.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events695.exact178148RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound178147.bound, RecordedBoundRefines] <;> decide)
      (LeftBound178147.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 183626 .coefficient)
      LeftBound22631.bound (LeftBound22631.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22632RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22631.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22631.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound178147.bound LeftBound22631.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound178147.bound, LeftBound22631.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound178147.actual selector witness) * (LeftBound22631.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound183627

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
