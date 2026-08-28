import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard114
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard115
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard779
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard782
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard825

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound125532
def owner : Owner := ⟨.program ⟨257⟩, ⟨57048⟩⟩
def transferEvent : Nat := 125532
def frameStart : Nat := 125436
def rule : BoundRule := .sum [.predecessor 0 125530 .coefficient, .predecessor 1 125531 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 125530 .coefficient)
      LeftAuthority125528.bound (LeftAuthority125528.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events490.exact125529RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority125528.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority125528.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 125531 .coefficient)
      LeftBound125524.bound (LeftBound125524.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events490.exact125526RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound125524.bound, RecordedBoundRefines] <;> decide)
      (LeftBound125524.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority125528.bound, LeftBound125524.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority125528.bound, LeftBound125524.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority125528.actual selector witness, LeftBound125524.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound125532

namespace LeftBound125536
def owner : Owner := ⟨.program ⟨257⟩, ⟨58793⟩⟩
def transferEvent : Nat := 125536
def frameStart : Nat := 125436
def rule : BoundRule := .sum [.predecessor 0 125534 .coefficient, .predecessor 1 125535 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 125534 .coefficient)
      LeftBound125532.bound (LeftBound125532.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events490.exact125533RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound125532.bound, RecordedBoundRefines] <;> decide)
      (LeftBound125532.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 125535 .coefficient)
      LeftBound125513.bound (LeftBound125513.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events490.exact125518RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound125513.bound, RecordedBoundRefines] <;> decide)
      (LeftBound125513.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound125532.bound, LeftBound125513.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound125532.bound, LeftBound125513.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound125532.actual selector witness, LeftBound125513.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound125536

namespace LeftBound125549
def owner : Owner := ⟨.program ⟨257⟩, ⟨58791⟩⟩
def transferEvent : Nat := 125549
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 125547 .coefficient, .predecessor 1 125548 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 125547 .coefficient)
      LeftBound125378.bound (LeftBound125378.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events490.exact125546RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound125378.bound, RecordedBoundRefines] <;> decide)
      (LeftBound125378.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 125548 .coefficient)
      LeftBound125361.bound (LeftBound125361.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events489.exact125368RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound125361.bound, RecordedBoundRefines] <;> decide)
      (LeftBound125361.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound125378.bound, LeftBound125361.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound125378.bound, LeftBound125361.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound125378.actual selector witness, LeftBound125361.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound125549

namespace LeftBound125552
def owner : Owner := ⟨.program ⟨257⟩, ⟨58791⟩⟩
def transferEvent : Nat := 125552
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 125546 .summary, .result 125368 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 125546 .summary)
      LeftBound125380.bound (LeftBound125380.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨57639⟩⟩) (rawTerms := some (Proof.Events490.exact125546RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound125380.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 125368 .summary)
      LeftBound125363.bound (LeftBound125363.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58790⟩⟩) (rawTerms := some (Proof.Events489.exact125368RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound125363.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound125380.bound, LeftBound125363.bound]
def bound : CoeffClass := .finite ⟨32190182365603518530196853751808, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound125380.bound, LeftBound125363.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound125380.actual selector witness, LeftBound125363.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound125552

namespace LeftBound125576
def owner : Owner := ⟨.program ⟨257⟩, ⟨24723⟩⟩
def transferEvent : Nat := 125576
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 125574 .coefficient) (.predecessor 1 125575 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 125574 .coefficient)
      LeftAuthority5605.bound (LeftAuthority5605.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5606RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5605.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5605.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 125575 .coefficient)
      LeftBound119776.bound (LeftBound119776.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events467.exact119778RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound119776.bound, RecordedBoundRefines] <;> decide)
      (LeftBound119776.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority5605.bound LeftBound119776.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5605.bound, LeftBound119776.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority5605.actual selector witness) * (LeftBound119776.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound125576

namespace LeftBound125581
def owner : Owner := ⟨.program ⟨257⟩, ⟨8122⟩⟩
def transferEvent : Nat := 125581
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 125579 .coefficient) (.predecessor 1 125580 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 125579 .coefficient)
      LeftBound119647.bound (LeftBound119647.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events467.exact119648RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound119647.bound, RecordedBoundRefines] <;> decide)
      (LeftBound119647.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 125580 .coefficient)
      LeftBound23091.bound (LeftBound23091.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23092RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23091.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23091.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound119647.bound LeftBound23091.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119647.bound, LeftBound23091.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound119647.actual selector witness) * (LeftBound23091.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound125581

namespace LeftBound125586
def owner : Owner := ⟨.program ⟨257⟩, ⟨24724⟩⟩
def transferEvent : Nat := 125586
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 125584 .coefficient, .predecessor 1 125585 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 125584 .coefficient)
      LeftBound125581.bound (LeftBound125581.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events490.exact125583RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound125581.bound, RecordedBoundRefines] <;> decide)
      (LeftBound125581.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 125585 .coefficient)
      LeftBound125576.bound (LeftBound125576.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events490.exact125578RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound125576.bound, RecordedBoundRefines] <;> decide)
      (LeftBound125576.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound125581.bound, LeftBound125576.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound125581.bound, LeftBound125576.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound125581.actual selector witness, LeftBound125576.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound125586

namespace LeftBound125590
def owner : Owner := ⟨.program ⟨257⟩, ⟨24725⟩⟩
def transferEvent : Nat := 125590
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 125588 .coefficient, .predecessor 1 125589 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 125588 .coefficient)
      LeftBound125586.bound (LeftBound125586.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events490.exact125587RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound125586.bound, RecordedBoundRefines] <;> decide)
      (LeftBound125586.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 125589 .coefficient)
      LeftBound23083.bound (LeftBound23083.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23084RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23083.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23083.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound125586.bound, LeftBound23083.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound125586.bound, LeftBound23083.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound125586.actual selector witness, LeftBound23083.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound125590

namespace LeftBound125591
def owner : Owner := ⟨.program ⟨257⟩, ⟨24725⟩⟩
def transferEvent : Nat := 125591
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨98⟩⟩]⟩ [⟨.result 23084 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 23084 .coefficient)
      LeftBound23083.bound (LeftBound23083.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨98⟩⟩) (rawTerms := some (Proof.Events090.exact23084RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23083.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23083.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound23083.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23083.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound23083.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound125591

namespace LeftBound125596
def owner : Owner := ⟨.program ⟨257⟩, ⟨53420⟩⟩
def transferEvent : Nat := 125596
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 125594 .coefficient) (.predecessor 1 125595 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 125594 .coefficient)
      LeftBound125590.bound (LeftBound125590.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events490.exact125593RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound125590.bound, RecordedBoundRefines] <;> decide)
      (LeftBound125590.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 125595 .coefficient)
      LeftAuthority5608.bound (LeftAuthority5608.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5609RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5608.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5608.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound125590.bound LeftAuthority5608.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound125590.bound, LeftAuthority5608.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound125590.actual selector witness) * (LeftAuthority5608.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound125596

namespace LeftBound125597
def owner : Owner := ⟨.program ⟨257⟩, ⟨53420⟩⟩
def transferEvent : Nat := 125597
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨53417⟩⟩], []⟩ [⟨.result 5609 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 5609 .coefficient)
      LeftAuthority5608.bound (LeftAuthority5608.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨53417⟩⟩) (rawTerms := some (Proof.Events021.exact5609RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5608.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5608.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5608.bound []
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5608.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority5608.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound125597

namespace LeftBound125598
def owner : Owner := ⟨.program ⟨257⟩, ⟨53420⟩⟩
def transferEvent : Nat := 125598
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 125593 .summary) (.transfer 125597) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 125593 .summary)
      LeftBound125591.bound (LeftBound125591.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨24725⟩⟩) (rawTerms := some (Proof.Events490.exact125593RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound125591.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 125597)
      LeftBound125597.bound (LeftBound125597.actual selector witness) := by
  exact .transfer (LeftBound125597.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound125591.bound LeftBound125597.bound
def bound : CoeffClass := .finite ⟨10223616, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound125591.bound, LeftBound125597.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound125591.actual selector witness) * (LeftBound125597.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound125598

namespace LeftBound125604
def owner : Owner := ⟨.program ⟨257⟩, ⟨53421⟩⟩
def transferEvent : Nat := 125604
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 125602 .coefficient) (.predecessor 1 125603 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 125602 .coefficient)
      LeftAuthority5608.bound (LeftAuthority5608.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5609RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5608.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5608.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 125603 .coefficient)
      LeftBound119776.bound (LeftBound119776.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events467.exact119778RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound119776.bound, RecordedBoundRefines] <;> decide)
      (LeftBound119776.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority5608.bound LeftBound119776.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5608.bound, LeftBound119776.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority5608.actual selector witness) * (LeftBound119776.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound125604

namespace LeftBound125609
def owner : Owner := ⟨.program ⟨257⟩, ⟨8139⟩⟩
def transferEvent : Nat := 125609
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 125607 .coefficient) (.predecessor 1 125608 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 125607 .coefficient)
      LeftBound119647.bound (LeftBound119647.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events467.exact119648RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound119647.bound, RecordedBoundRefines] <;> decide)
      (LeftBound119647.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 125608 .coefficient)
      LeftBound23132.bound (LeftBound23132.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23133RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23132.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23132.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound119647.bound LeftBound23132.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119647.bound, LeftBound23132.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound119647.actual selector witness) * (LeftBound23132.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound125609

namespace LeftBound125614
def owner : Owner := ⟨.program ⟨257⟩, ⟨53422⟩⟩
def transferEvent : Nat := 125614
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 125612 .coefficient, .predecessor 1 125613 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 125612 .coefficient)
      LeftBound125609.bound (LeftBound125609.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events490.exact125611RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound125609.bound, RecordedBoundRefines] <;> decide)
      (LeftBound125609.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 125613 .coefficient)
      LeftBound125604.bound (LeftBound125604.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events490.exact125606RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound125604.bound, RecordedBoundRefines] <;> decide)
      (LeftBound125604.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound125609.bound, LeftBound125604.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound125609.bound, LeftBound125604.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound125609.actual selector witness, LeftBound125604.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound125614

namespace LeftBound125618
def owner : Owner := ⟨.program ⟨257⟩, ⟨53423⟩⟩
def transferEvent : Nat := 125618
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 125616 .coefficient, .predecessor 1 125617 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 125616 .coefficient)
      LeftBound125614.bound (LeftBound125614.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events490.exact125615RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound125614.bound, RecordedBoundRefines] <;> decide)
      (LeftBound125614.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 125617 .coefficient)
      LeftBound23124.bound (LeftBound23124.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23125RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23124.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23124.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound125614.bound, LeftBound23124.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound125614.bound, LeftBound23124.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound125614.actual selector witness, LeftBound23124.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound125618

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
