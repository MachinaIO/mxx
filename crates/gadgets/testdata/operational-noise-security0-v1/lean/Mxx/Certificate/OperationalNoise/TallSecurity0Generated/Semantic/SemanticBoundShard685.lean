import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound99669
def owner : Owner := ⟨.program ⟨214⟩, ⟨19375⟩⟩
def transferEvent : Nat := 99669
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 99667 .coefficient) (.value (.predecessor 1 99668 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99667 .coefficient)
      LeftAuthority99665.bound (LeftAuthority99665.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events389.exact99666RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99665.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99665.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99668 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority99665.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority99665.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority99665.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound99669

namespace LeftBound99673
def owner : Owner := ⟨.program ⟨214⟩, ⟨19376⟩⟩
def transferEvent : Nat := 99673
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 99671 .coefficient) (.predecessor 1 99672 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99671 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99672 .coefficient)
      LeftBound99669.bound (LeftBound99669.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events389.exact99670RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99669.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99669.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound99669.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound99669.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound99669.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound99673

namespace LeftBound99674
def owner : Owner := ⟨.program ⟨214⟩, ⟨19376⟩⟩
def transferEvent : Nat := 99674
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19373⟩⟩]⟩ [⟨.result 99666 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 99666 .coefficient)
      LeftAuthority99665.bound (LeftAuthority99665.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19373⟩⟩) (rawTerms := some (Proof.Events389.exact99666RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99665.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99665.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority99665.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority99665.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority99665.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound99674

namespace LeftBound99675
def owner : Owner := ⟨.program ⟨214⟩, ⟨19376⟩⟩
def transferEvent : Nat := 99675
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 99674) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 99674)
      LeftBound99674.bound (LeftBound99674.actual selector witness) := by
  exact .transfer (LeftBound99674.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound99674.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound99674.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound99674.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound99675

namespace LeftBound99730
def owner : Owner := ⟨.program ⟨214⟩, ⟨13747⟩⟩
def transferEvent : Nat := 99730
def frameStart : Nat := 99713
def rule : BoundRule := .product (.predecessor 0 99728 .coefficient) (.predecessor 1 99729 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99728 .coefficient)
      LeftAuthority99726.bound (LeftAuthority99726.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events389.exact99727RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99726.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99726.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99729 .coefficient)
      LeftAuthority99723.bound (LeftAuthority99723.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events389.exact99724RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99723.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99723.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority99726.bound LeftAuthority99723.bound
def bound : CoeffClass := .finite ⟨144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority99726.bound, LeftAuthority99723.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority99726.actual selector witness) * (LeftAuthority99723.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound99730

namespace LeftBound99734
def owner : Owner := ⟨.program ⟨214⟩, ⟨13748⟩⟩
def transferEvent : Nat := 99734
def frameStart : Nat := 99713
def rule : BoundRule := .identity (.predecessor 0 99733 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99733 .coefficient)
      LeftBound99730.bound (LeftBound99730.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events389.exact99732RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99730.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99730.derived selector witness)

def rawBound : CoeffClass := LeftBound99730.bound
def bound : CoeffClass := .finite ⟨144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99730.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound99730.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound99734

namespace LeftBound99751
def owner : Owner := ⟨.program ⟨214⟩, ⟨13872⟩⟩
def transferEvent : Nat := 99751
def frameStart : Nat := 99713
def rule : BoundRule := .sum [.predecessor 0 99749 .coefficient, .predecessor 1 99750 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99749 .coefficient)
      LeftBound99734.bound (LeftBound99734.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound99734.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99750 .coefficient)
      LeftAuthority99747.bound (LeftAuthority99747.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority99747.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99734.bound, LeftAuthority99747.bound]
def bound : CoeffClass := .finite ⟨144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99734.bound, LeftAuthority99747.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound99734.actual selector witness, LeftAuthority99747.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99751

namespace LeftBound99754
def owner : Owner := ⟨.program ⟨214⟩, ⟨13873⟩⟩
def transferEvent : Nat := 99754
def frameStart : Nat := 99713
def rule : BoundRule := .identity (.predecessor 0 99753 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99753 .coefficient)
      LeftBound99751.bound (LeftBound99751.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound99751.derived selector witness)

def rawBound : CoeffClass := LeftBound99751.bound
def bound : CoeffClass := .finite ⟨144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99751.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound99751.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound99754

namespace LeftBound99760
def owner : Owner := ⟨.program ⟨214⟩, ⟨13874⟩⟩
def transferEvent : Nat := 99760
def frameStart : Nat := 99713
def rule : BoundRule := .product (.predecessor 0 99758 .coefficient) (.predecessor 1 99759 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99758 .coefficient)
      LeftAuthority99756.bound (LeftAuthority99756.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events389.exact99757RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99756.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99756.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99759 .coefficient)
      LeftBound99754.bound (LeftBound99754.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events389.exact99755RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99754.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99754.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority99756.bound LeftBound99754.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority99756.bound, LeftBound99754.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority99756.actual selector witness) * (LeftBound99754.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound99760

namespace LeftBound99776
def owner : Owner := ⟨.program ⟨214⟩, ⟨7847⟩⟩
def transferEvent : Nat := 99776
def frameStart : Nat := 99713
def rule : BoundRule := .scale (.predecessor 0 99774 .coefficient) (.value (.predecessor 1 99775 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99774 .coefficient)
      LeftAuthority99772.bound (LeftAuthority99772.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events389.exact99773RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99772.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99772.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99775 .coefficient)
      LeftAuthority99763.bound (LeftAuthority99763.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority99763.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority99772.bound LeftAuthority99763.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority99772.bound, LeftAuthority99763.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority99772.actual selector witness) * (LeftAuthority99763.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound99776

namespace LeftBound99779
def owner : Owner := ⟨.program ⟨214⟩, ⟨6794⟩⟩
def transferEvent : Nat := 99779
def frameStart : Nat := 99713
def rule : BoundRule := .identity (.predecessor 0 99778 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99778 .coefficient)
      LeftAuthority99766.bound (LeftAuthority99766.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events389.exact99767RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99766.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99766.derived selector witness)

def rawBound : CoeffClass := LeftAuthority99766.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority99766.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority99766.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound99779

namespace LeftBound99783
def owner : Owner := ⟨.program ⟨214⟩, ⟨7848⟩⟩
def transferEvent : Nat := 99783
def frameStart : Nat := 99713
def rule : BoundRule := .product (.predecessor 0 99781 .coefficient) (.predecessor 1 99782 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99781 .coefficient)
      LeftBound99779.bound (LeftBound99779.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events389.exact99780RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99779.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99779.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99782 .coefficient)
      LeftBound99776.bound (LeftBound99776.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events389.exact99777RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99776.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99776.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound99779.bound LeftBound99776.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99779.bound, LeftBound99776.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound99779.actual selector witness) * (LeftBound99776.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound99783

namespace LeftBound99788
def owner : Owner := ⟨.program ⟨214⟩, ⟨13875⟩⟩
def transferEvent : Nat := 99788
def frameStart : Nat := 99713
def rule : BoundRule := .sum [.predecessor 0 99786 .coefficient, .predecessor 1 99787 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99786 .coefficient)
      LeftBound99783.bound (LeftBound99783.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events389.exact99785RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99783.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99783.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99787 .coefficient)
      LeftBound99760.bound (LeftBound99760.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events389.exact99762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99760.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99760.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99783.bound, LeftBound99760.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99783.bound, LeftBound99760.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound99783.actual selector witness, LeftBound99760.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99788

namespace LeftBound99792
def owner : Owner := ⟨.program ⟨214⟩, ⟨25902⟩⟩
def transferEvent : Nat := 99792
def frameStart : Nat := 99713
def rule : BoundRule := .product (.predecessor 0 99790 .coefficient) (.predecessor 1 99791 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99790 .coefficient)
      LeftBound99788.bound (LeftBound99788.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events389.exact99789RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99788.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99788.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99791 .coefficient)
      LeftAuthority99745.bound (LeftAuthority99745.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events389.exact99746RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99745.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99745.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound99788.bound LeftAuthority99745.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99788.bound, LeftAuthority99745.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound99788.actual selector witness) * (LeftAuthority99745.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound99792

namespace LeftBound99803
def owner : Owner := ⟨.program ⟨214⟩, ⟨15694⟩⟩
def transferEvent : Nat := 99803
def frameStart : Nat := 99713
def rule : BoundRule := .product (.predecessor 0 99801 .coefficient) (.predecessor 1 99802 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99801 .coefficient)
      LeftAuthority99756.bound (LeftAuthority99756.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events389.exact99757RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99756.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99756.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99802 .coefficient)
      LeftAuthority99799.bound (LeftAuthority99799.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events389.exact99800RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99799.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99799.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority99756.bound LeftAuthority99799.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority99756.bound, LeftAuthority99799.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority99756.actual selector witness) * (LeftAuthority99799.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound99803

namespace LeftBound99811
def owner : Owner := ⟨.program ⟨214⟩, ⟨15695⟩⟩
def transferEvent : Nat := 99811
def frameStart : Nat := 99713
def rule : BoundRule := .sum [.predecessor 0 99809 .coefficient, .predecessor 1 99810 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99809 .coefficient)
      LeftAuthority99807.bound (LeftAuthority99807.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events389.exact99808RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99807.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99807.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99810 .coefficient)
      LeftBound99803.bound (LeftBound99803.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events389.exact99805RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99803.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99803.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority99807.bound, LeftBound99803.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority99807.bound, LeftBound99803.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority99807.actual selector witness, LeftBound99803.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99811

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
