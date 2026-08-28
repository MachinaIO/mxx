import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard261
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard315

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound47609
def owner : Owner := ⟨.program ⟨214⟩, ⟨28974⟩⟩
def transferEvent : Nat := 47609
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 47604 .summary) (.transfer 47608) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 47604 .summary)
      LeftBound47603.bound (LeftBound47603.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28973⟩⟩) (rawTerms := some (Proof.Events185.exact47604RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound47603.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 47608)
      LeftBound47608.bound (LeftBound47608.actual selector witness) := by
  exact .transfer (LeftBound47608.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound47603.bound LeftBound47608.bound
def bound : CoeffClass := .finite ⟨4742816766803936246568583168, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound47603.bound, LeftBound47608.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound47603.actual selector witness) * (LeftBound47608.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound47609

namespace LeftBound47624
def owner : Owner := ⟨.program ⟨214⟩, ⟨28755⟩⟩
def transferEvent : Nat := 47624
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 47622 .coefficient) (.predecessor 1 47623 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47622 .coefficient)
      LeftBound39211.bound (LeftBound39211.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39215RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39211.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39211.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47623 .coefficient)
      LeftAuthority47620.bound (LeftAuthority47620.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events186.exact47621RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47620.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47620.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound39211.bound LeftAuthority47620.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39211.bound, LeftAuthority47620.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound39211.actual selector witness) * (LeftAuthority47620.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound47624

namespace LeftBound47625
def owner : Owner := ⟨.program ⟨214⟩, ⟨28755⟩⟩
def transferEvent : Nat := 47625
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28753⟩⟩]⟩ [⟨.result 47621 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 47621 .coefficient)
      LeftAuthority47620.bound (LeftAuthority47620.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28753⟩⟩) (rawTerms := some (Proof.Events186.exact47621RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47620.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47620.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority47620.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority47620.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority47620.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound47625

namespace LeftBound47626
def owner : Owner := ⟨.program ⟨214⟩, ⟨28755⟩⟩
def transferEvent : Nat := 47626
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 39215 .summary) (.transfer 47625) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 39215 .summary)
      LeftBound39214.bound (LeftBound39214.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25231⟩⟩) (rawTerms := some (Proof.Events153.exact39215RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound39214.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 47625)
      LeftBound47625.bound (LeftBound47625.actual selector witness) := by
  exact .transfer (LeftBound47625.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound39214.bound LeftBound47625.bound
def bound : CoeffClass := .finite ⟨1292270184133468094464, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39214.bound, LeftBound47625.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound39214.actual selector witness) * (LeftBound47625.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound47626

namespace LeftBound47637
def owner : Owner := ⟨.program ⟨214⟩, ⟨21914⟩⟩
def transferEvent : Nat := 47637
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 47635 .coefficient) (.value (.predecessor 1 47636 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47635 .coefficient)
      LeftAuthority47633.bound (LeftAuthority47633.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events186.exact47634RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47633.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47633.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47636 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority47633.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority47633.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority47633.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound47637

namespace LeftBound47641
def owner : Owner := ⟨.program ⟨214⟩, ⟨21915⟩⟩
def transferEvent : Nat := 47641
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 47639 .coefficient) (.predecessor 1 47640 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47639 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47640 .coefficient)
      LeftBound47637.bound (LeftBound47637.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events186.exact47638RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47637.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47637.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound47637.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound47637.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound47637.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound47641

namespace LeftBound47642
def owner : Owner := ⟨.program ⟨214⟩, ⟨21915⟩⟩
def transferEvent : Nat := 47642
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21912⟩⟩]⟩ [⟨.result 47634 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 47634 .coefficient)
      LeftAuthority47633.bound (LeftAuthority47633.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21912⟩⟩) (rawTerms := some (Proof.Events186.exact47634RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47633.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47633.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority47633.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority47633.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority47633.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound47642

namespace LeftBound47643
def owner : Owner := ⟨.program ⟨214⟩, ⟨21915⟩⟩
def transferEvent : Nat := 47643
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 47642) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 47642)
      LeftBound47642.bound (LeftBound47642.actual selector witness) := by
  exact .transfer (LeftBound47642.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound47642.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound47642.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound47642.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound47643

namespace LeftBound47738
def owner : Owner := ⟨.program ⟨214⟩, ⟨16390⟩⟩
def transferEvent : Nat := 47738
def frameStart : Nat := 47699
def rule : BoundRule := .identity (.predecessor 0 47737 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47737 .coefficient)
      LeftAuthority47735.bound (LeftAuthority47735.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events186.exact47736RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47735.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47735.derived selector witness)

def rawBound : CoeffClass := LeftAuthority47735.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority47735.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority47735.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound47738

namespace LeftBound47755
def owner : Owner := ⟨.program ⟨214⟩, ⟨16429⟩⟩
def transferEvent : Nat := 47755
def frameStart : Nat := 47699
def rule : BoundRule := .sum [.predecessor 0 47753 .coefficient, .predecessor 1 47754 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47753 .coefficient)
      LeftBound47738.bound (LeftBound47738.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound47738.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47754 .coefficient)
      LeftAuthority47751.bound (LeftAuthority47751.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority47751.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound47738.bound, LeftAuthority47751.bound]
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound47738.bound, LeftAuthority47751.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound47738.actual selector witness, LeftAuthority47751.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound47755

namespace LeftBound47758
def owner : Owner := ⟨.program ⟨214⟩, ⟨16430⟩⟩
def transferEvent : Nat := 47758
def frameStart : Nat := 47699
def rule : BoundRule := .identity (.predecessor 0 47757 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47757 .coefficient)
      LeftBound47755.bound (LeftBound47755.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound47755.derived selector witness)

def rawBound : CoeffClass := LeftBound47755.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound47755.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound47755.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound47758

namespace LeftBound47764
def owner : Owner := ⟨.program ⟨214⟩, ⟨16431⟩⟩
def transferEvent : Nat := 47764
def frameStart : Nat := 47699
def rule : BoundRule := .product (.predecessor 0 47762 .coefficient) (.predecessor 1 47763 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47762 .coefficient)
      LeftAuthority47760.bound (LeftAuthority47760.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events186.exact47761RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47760.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47763 .coefficient)
      LeftBound47758.bound (LeftBound47758.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events186.exact47759RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47758.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47758.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority47760.bound LeftBound47758.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority47760.bound, LeftBound47758.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority47760.actual selector witness) * (LeftBound47758.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound47764

namespace LeftBound47772
def owner : Owner := ⟨.program ⟨214⟩, ⟨16432⟩⟩
def transferEvent : Nat := 47772
def frameStart : Nat := 47699
def rule : BoundRule := .sum [.predecessor 0 47770 .coefficient, .predecessor 1 47771 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47770 .coefficient)
      LeftAuthority47768.bound (LeftAuthority47768.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events186.exact47769RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47768.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47768.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47771 .coefficient)
      LeftBound47764.bound (LeftBound47764.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events186.exact47766RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47764.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47764.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority47768.bound, LeftBound47764.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority47768.bound, LeftBound47764.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority47768.actual selector witness, LeftBound47764.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound47772

namespace LeftBound47776
def owner : Owner := ⟨.program ⟨214⟩, ⟨28754⟩⟩
def transferEvent : Nat := 47776
def frameStart : Nat := 47699
def rule : BoundRule := .product (.predecessor 0 47774 .coefficient) (.predecessor 1 47775 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47774 .coefficient)
      LeftBound47772.bound (LeftBound47772.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events186.exact47773RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47772.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47772.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47775 .coefficient)
      LeftAuthority47749.bound (LeftAuthority47749.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events186.exact47750RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47749.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47749.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound47772.bound LeftAuthority47749.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound47772.bound, LeftAuthority47749.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound47772.actual selector witness) * (LeftAuthority47749.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound47776

namespace LeftBound47787
def owner : Owner := ⟨.program ⟨214⟩, ⟨18872⟩⟩
def transferEvent : Nat := 47787
def frameStart : Nat := 47699
def rule : BoundRule := .product (.predecessor 0 47785 .coefficient) (.predecessor 1 47786 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47785 .coefficient)
      LeftAuthority47760.bound (LeftAuthority47760.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events186.exact47761RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47760.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47786 .coefficient)
      LeftAuthority47783.bound (LeftAuthority47783.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events186.exact47784RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47783.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47783.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority47760.bound LeftAuthority47783.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority47760.bound, LeftAuthority47783.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority47760.actual selector witness) * (LeftAuthority47783.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound47787

namespace LeftBound47795
def owner : Owner := ⟨.program ⟨214⟩, ⟨18877⟩⟩
def transferEvent : Nat := 47795
def frameStart : Nat := 47699
def rule : BoundRule := .sum [.predecessor 0 47793 .coefficient, .predecessor 1 47794 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47793 .coefficient)
      LeftAuthority47791.bound (LeftAuthority47791.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events186.exact47792RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47791.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47791.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47794 .coefficient)
      LeftBound47787.bound (LeftBound47787.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events186.exact47789RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47787.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47787.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority47791.bound, LeftBound47787.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority47791.bound, LeftBound47787.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority47791.actual selector witness, LeftBound47787.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound47795

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
