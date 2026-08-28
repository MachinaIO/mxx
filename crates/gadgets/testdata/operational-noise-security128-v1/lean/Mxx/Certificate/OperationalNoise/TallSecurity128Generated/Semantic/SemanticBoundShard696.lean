import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard082
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard083
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard678
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard681
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard695

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound107109
def owner : Owner := ⟨.program ⟨257⟩, ⟨37143⟩⟩
def transferEvent : Nat := 107109
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 107107 .coefficient, .predecessor 1 107108 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 107107 .coefficient)
      LeftBound107105.bound (LeftBound107105.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107106RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107105.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107105.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 107108 .coefficient)
      LeftBound19075.bound (LeftBound19075.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact19076RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19075.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19075.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107105.bound, LeftBound19075.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107105.bound, LeftBound19075.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound107105.actual selector witness, LeftBound19075.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107109

namespace LeftBound107110
def owner : Owner := ⟨.program ⟨257⟩, ⟨37143⟩⟩
def transferEvent : Nat := 107110
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨107⟩⟩]⟩ [⟨.result 19076 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 19076 .coefficient)
      LeftBound19075.bound (LeftBound19075.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨107⟩⟩) (rawTerms := some (Proof.Events074.exact19076RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19075.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19075.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound19075.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19075.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound19075.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound107110

namespace LeftBound107115
def owner : Owner := ⟨.program ⟨257⟩, ⟨37144⟩⟩
def transferEvent : Nat := 107115
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 107113 .coefficient) (.predecessor 1 107114 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 107113 .coefficient)
      LeftBound107109.bound (LeftBound107109.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107112RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107109.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107109.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 107114 .coefficient)
      LeftAuthority4676.bound (LeftAuthority4676.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events018.exact4677RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4676.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4676.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound107109.bound LeftAuthority4676.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107109.bound, LeftAuthority4676.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound107109.actual selector witness) * (LeftAuthority4676.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound107115

namespace LeftBound107116
def owner : Owner := ⟨.program ⟨257⟩, ⟨37144⟩⟩
def transferEvent : Nat := 107116
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨13896⟩⟩], []⟩ [⟨.result 4677 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 4677 .coefficient)
      LeftAuthority4676.bound (LeftAuthority4676.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨13896⟩⟩) (rawTerms := some (Proof.Events018.exact4677RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4676.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4676.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority4676.bound []
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4676.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority4676.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound107116

namespace LeftBound107117
def owner : Owner := ⟨.program ⟨257⟩, ⟨37144⟩⟩
def transferEvent : Nat := 107117
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 107112 .summary) (.transfer 107116) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 107112 .summary)
      LeftBound107110.bound (LeftBound107110.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨37143⟩⟩) (rawTerms := some (Proof.Events418.exact107112RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound107110.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 107116)
      LeftBound107116.bound (LeftBound107116.actual selector witness) := by
  exact .transfer (LeftBound107116.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound107110.bound LeftBound107116.bound
def bound : CoeffClass := .finite ⟨35782656, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107110.bound, LeftBound107116.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound107110.actual selector witness) * (LeftBound107116.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound107117

namespace LeftBound107123
def owner : Owner := ⟨.program ⟨257⟩, ⟨13897⟩⟩
def transferEvent : Nat := 107123
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 107121 .coefficient) (.predecessor 1 107122 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 107121 .coefficient)
      LeftAuthority4676.bound (LeftAuthority4676.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events018.exact4677RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4676.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4676.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 107122 .coefficient)
      LeftBound105151.bound (LeftBound105151.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events410.exact105153RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105151.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105151.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority4676.bound LeftBound105151.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4676.bound, LeftBound105151.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority4676.actual selector witness) * (LeftBound105151.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound107123

namespace LeftBound107128
def owner : Owner := ⟨.program ⟨257⟩, ⟨8718⟩⟩
def transferEvent : Nat := 107128
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 107126 .coefficient) (.predecessor 1 107127 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 107126 .coefficient)
      LeftBound105022.bound (LeftBound105022.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events410.exact105023RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105022.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105022.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 107127 .coefficient)
      LeftBound19124.bound (LeftBound19124.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact19125RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19124.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19124.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound105022.bound LeftBound19124.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105022.bound, LeftBound19124.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound105022.actual selector witness) * (LeftBound19124.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound107128

namespace LeftBound107133
def owner : Owner := ⟨.program ⟨257⟩, ⟨13898⟩⟩
def transferEvent : Nat := 107133
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 107131 .coefficient, .predecessor 1 107132 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 107131 .coefficient)
      LeftBound107128.bound (LeftBound107128.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107130RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107128.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107128.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 107132 .coefficient)
      LeftBound107123.bound (LeftBound107123.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107125RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107123.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107123.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107128.bound, LeftBound107123.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107128.bound, LeftBound107123.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound107128.actual selector witness, LeftBound107123.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107133

namespace LeftBound107137
def owner : Owner := ⟨.program ⟨257⟩, ⟨13899⟩⟩
def transferEvent : Nat := 107137
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 107135 .coefficient, .predecessor 1 107136 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 107135 .coefficient)
      LeftBound107133.bound (LeftBound107133.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107134RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107133.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107133.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 107136 .coefficient)
      LeftBound19116.bound (LeftBound19116.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact19117RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19116.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19116.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107133.bound, LeftBound19116.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107133.bound, LeftBound19116.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound107133.actual selector witness, LeftBound19116.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107137

namespace LeftBound107138
def owner : Owner := ⟨.program ⟨257⟩, ⟨13899⟩⟩
def transferEvent : Nat := 107138
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨124⟩⟩]⟩ [⟨.result 19117 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 19117 .coefficient)
      LeftBound19116.bound (LeftBound19116.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨124⟩⟩) (rawTerms := some (Proof.Events074.exact19117RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19116.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19116.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound19116.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19116.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound19116.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound107138

namespace LeftBound107143
def owner : Owner := ⟨.program ⟨257⟩, ⟨13900⟩⟩
def transferEvent : Nat := 107143
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 107141 .coefficient) (.predecessor 1 107142 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 107141 .coefficient)
      LeftBound107137.bound (LeftBound107137.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107140RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107137.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107137.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 107142 .coefficient)
      LeftBound19113.bound (LeftBound19113.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact19114RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19113.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19113.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound107137.bound LeftBound19113.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107137.bound, LeftBound19113.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound107137.actual selector witness) * (LeftBound19113.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound107143

namespace LeftBound107144
def owner : Owner := ⟨.program ⟨257⟩, ⟨13900⟩⟩
def transferEvent : Nat := 107144
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩ [⟨.result 19110 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 19110 .coefficient)
      LeftAuthority19109.bound (LeftAuthority19109.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨9553⟩⟩) (rawTerms := some (Proof.Events074.exact19110RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19109.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19109.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority19109.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19109.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority19109.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound107144

namespace LeftBound107145
def owner : Owner := ⟨.program ⟨257⟩, ⟨13900⟩⟩
def transferEvent : Nat := 107145
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 107140 .summary) (.transfer 107144) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 107140 .summary)
      LeftBound107138.bound (LeftBound107138.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨13899⟩⟩) (rawTerms := some (Proof.Events418.exact107140RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound107138.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 107144)
      LeftBound107144.bound (LeftBound107144.actual selector witness) := by
  exact .transfer (LeftBound107144.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound107138.bound LeftBound107144.bound
def bound : CoeffClass := .finite ⟨279172874240, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107138.bound, LeftBound107144.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound107138.actual selector witness) * (LeftBound107144.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound107145

namespace LeftBound107153
def owner : Owner := ⟨.program ⟨257⟩, ⟨37145⟩⟩
def transferEvent : Nat := 107153
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 107151 .coefficient, .predecessor 1 107152 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 107151 .coefficient)
      LeftBound107143.bound (LeftBound107143.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107150RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107143.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107143.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 107152 .coefficient)
      LeftBound107115.bound (LeftBound107115.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107120RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107115.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107115.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107143.bound, LeftBound107115.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107143.bound, LeftBound107115.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound107143.actual selector witness, LeftBound107115.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107153

namespace LeftBound107155
def owner : Owner := ⟨.program ⟨257⟩, ⟨37145⟩⟩
def transferEvent : Nat := 107155
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 107150 .summary, .result 107120 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 107150 .summary)
      LeftBound107145.bound (LeftBound107145.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨13900⟩⟩) (rawTerms := some (Proof.Events418.exact107150RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound107145.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 107120 .summary)
      LeftBound107117.bound (LeftBound107117.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨37144⟩⟩) (rawTerms := some (Proof.Events418.exact107120RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound107117.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107145.bound, LeftBound107117.bound]
def bound : CoeffClass := .finite ⟨279208656896, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107145.bound, LeftBound107117.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound107145.actual selector witness, LeftBound107117.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107155

namespace LeftBound107159
def owner : Owner := ⟨.program ⟨257⟩, ⟨38951⟩⟩
def transferEvent : Nat := 107159
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 107157 .coefficient) (.predecessor 1 107158 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 107157 .coefficient)
      LeftBound107153.bound (LeftBound107153.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107156RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107153.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107153.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 107158 .coefficient)
      LeftAuthority107091.bound (LeftAuthority107091.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107092RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority107091.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority107091.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound107153.bound LeftAuthority107091.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107153.bound, LeftAuthority107091.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound107153.actual selector witness) * (LeftAuthority107091.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound107159

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
