import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard257

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound38736
def owner : Owner := ⟨.program ⟨214⟩, ⟨28979⟩⟩
def transferEvent : Nat := 38736
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 38734 .coefficient) (.predecessor 1 38735 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38734 .coefficient)
      LeftBound38729.bound (LeftBound38729.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events151.exact38733RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38729.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38729.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38735 .coefficient)
      LeftAuthority38455.bound (LeftAuthority38455.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events150.exact38456RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38455.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38455.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound38729.bound LeftAuthority38455.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38729.bound, LeftAuthority38455.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound38729.actual selector witness) * (LeftAuthority38455.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38736

namespace LeftBound38737
def owner : Owner := ⟨.program ⟨214⟩, ⟨28979⟩⟩
def transferEvent : Nat := 38737
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28977⟩⟩]⟩ [⟨.result 38456 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 38456 .coefficient)
      LeftAuthority38455.bound (LeftAuthority38455.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28977⟩⟩) (rawTerms := some (Proof.Events150.exact38456RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38455.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38455.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority38455.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority38455.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority38455.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound38737

namespace LeftBound38738
def owner : Owner := ⟨.program ⟨214⟩, ⟨28979⟩⟩
def transferEvent : Nat := 38738
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 38733 .summary) (.transfer 38737) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 38733 .summary)
      LeftBound38732.bound (LeftBound38732.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25385⟩⟩) (rawTerms := some (Proof.Events151.exact38733RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound38732.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 38737)
      LeftBound38737.bound (LeftBound38737.actual selector witness) := by
  exact .transfer (LeftBound38737.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound38732.bound LeftBound38737.bound
def bound : CoeffClass := .finite ⟨1292315009023509266432, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38732.bound, LeftBound38737.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound38732.actual selector witness) * (LeftBound38737.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38738

namespace LeftBound38749
def owner : Owner := ⟨.program ⟨214⟩, ⟨22130⟩⟩
def transferEvent : Nat := 38749
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 38747 .coefficient) (.value (.predecessor 1 38748 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38747 .coefficient)
      LeftAuthority38745.bound (LeftAuthority38745.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events151.exact38746RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38745.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38745.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38748 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority38745.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority38745.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority38745.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound38749

namespace LeftBound38753
def owner : Owner := ⟨.program ⟨214⟩, ⟨22131⟩⟩
def transferEvent : Nat := 38753
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 38751 .coefficient) (.predecessor 1 38752 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38751 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38752 .coefficient)
      LeftBound38749.bound (LeftBound38749.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events151.exact38750RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38749.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38749.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound38749.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound38749.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound38749.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38753

namespace LeftBound38754
def owner : Owner := ⟨.program ⟨214⟩, ⟨22131⟩⟩
def transferEvent : Nat := 38754
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22128⟩⟩]⟩ [⟨.result 38746 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 38746 .coefficient)
      LeftAuthority38745.bound (LeftAuthority38745.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22128⟩⟩) (rawTerms := some (Proof.Events151.exact38746RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38745.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38745.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority38745.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority38745.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority38745.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound38754

namespace LeftBound38755
def owner : Owner := ⟨.program ⟨214⟩, ⟨22131⟩⟩
def transferEvent : Nat := 38755
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 38754) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 38754)
      LeftBound38754.bound (LeftBound38754.actual selector witness) := by
  exact .transfer (LeftBound38754.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound38754.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound38754.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound38754.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38755

namespace LeftBound38850
def owner : Owner := ⟨.program ⟨214⟩, ⟨16474⟩⟩
def transferEvent : Nat := 38850
def frameStart : Nat := 38811
def rule : BoundRule := .identity (.predecessor 0 38849 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38849 .coefficient)
      LeftAuthority38847.bound (LeftAuthority38847.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events151.exact38848RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38847.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38847.derived selector witness)

def rawBound : CoeffClass := LeftAuthority38847.bound
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority38847.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority38847.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound38850

namespace LeftBound38867
def owner : Owner := ⟨.program ⟨214⟩, ⟨16513⟩⟩
def transferEvent : Nat := 38867
def frameStart : Nat := 38811
def rule : BoundRule := .sum [.predecessor 0 38865 .coefficient, .predecessor 1 38866 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38865 .coefficient)
      LeftBound38850.bound (LeftBound38850.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound38850.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38866 .coefficient)
      LeftAuthority38863.bound (LeftAuthority38863.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority38863.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound38850.bound, LeftAuthority38863.bound]
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38850.bound, LeftAuthority38863.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound38850.actual selector witness, LeftAuthority38863.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound38867

namespace LeftBound38870
def owner : Owner := ⟨.program ⟨214⟩, ⟨16514⟩⟩
def transferEvent : Nat := 38870
def frameStart : Nat := 38811
def rule : BoundRule := .identity (.predecessor 0 38869 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38869 .coefficient)
      LeftBound38867.bound (LeftBound38867.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound38867.derived selector witness)

def rawBound : CoeffClass := LeftBound38867.bound
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38867.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound38867.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound38870

namespace LeftBound38876
def owner : Owner := ⟨.program ⟨214⟩, ⟨16515⟩⟩
def transferEvent : Nat := 38876
def frameStart : Nat := 38811
def rule : BoundRule := .product (.predecessor 0 38874 .coefficient) (.predecessor 1 38875 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38874 .coefficient)
      LeftAuthority38872.bound (LeftAuthority38872.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events151.exact38873RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38872.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38872.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38875 .coefficient)
      LeftBound38870.bound (LeftBound38870.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events151.exact38871RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38870.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38870.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority38872.bound LeftBound38870.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority38872.bound, LeftBound38870.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority38872.actual selector witness) * (LeftBound38870.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38876

namespace LeftBound38884
def owner : Owner := ⟨.program ⟨214⟩, ⟨16516⟩⟩
def transferEvent : Nat := 38884
def frameStart : Nat := 38811
def rule : BoundRule := .sum [.predecessor 0 38882 .coefficient, .predecessor 1 38883 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38882 .coefficient)
      LeftAuthority38880.bound (LeftAuthority38880.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events151.exact38881RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38880.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38880.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38883 .coefficient)
      LeftBound38876.bound (LeftBound38876.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events151.exact38878RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38876.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38876.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority38880.bound, LeftBound38876.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority38880.bound, LeftBound38876.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority38880.actual selector witness, LeftBound38876.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound38884

namespace LeftBound38888
def owner : Owner := ⟨.program ⟨214⟩, ⟨28978⟩⟩
def transferEvent : Nat := 38888
def frameStart : Nat := 38811
def rule : BoundRule := .product (.predecessor 0 38886 .coefficient) (.predecessor 1 38887 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38886 .coefficient)
      LeftBound38884.bound (LeftBound38884.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events151.exact38885RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38884.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38884.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38887 .coefficient)
      LeftAuthority38861.bound (LeftAuthority38861.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events151.exact38862RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38861.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38861.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound38884.bound LeftAuthority38861.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38884.bound, LeftAuthority38861.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound38884.actual selector witness) * (LeftAuthority38861.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38888

namespace LeftBound38899
def owner : Owner := ⟨.program ⟨214⟩, ⟨17911⟩⟩
def transferEvent : Nat := 38899
def frameStart : Nat := 38811
def rule : BoundRule := .product (.predecessor 0 38897 .coefficient) (.predecessor 1 38898 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38897 .coefficient)
      LeftAuthority38872.bound (LeftAuthority38872.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events151.exact38873RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38872.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38872.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38898 .coefficient)
      LeftAuthority38895.bound (LeftAuthority38895.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events151.exact38896RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38895.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38895.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority38872.bound LeftAuthority38895.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority38872.bound, LeftAuthority38895.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority38872.actual selector witness) * (LeftAuthority38895.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38899

namespace LeftBound38907
def owner : Owner := ⟨.program ⟨214⟩, ⟨17912⟩⟩
def transferEvent : Nat := 38907
def frameStart : Nat := 38811
def rule : BoundRule := .sum [.predecessor 0 38905 .coefficient, .predecessor 1 38906 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38905 .coefficient)
      LeftAuthority38903.bound (LeftAuthority38903.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events151.exact38904RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38903.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38903.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38906 .coefficient)
      LeftBound38899.bound (LeftBound38899.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events151.exact38901RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38899.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38899.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority38903.bound, LeftBound38899.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority38903.bound, LeftBound38899.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority38903.actual selector witness, LeftBound38899.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound38907

namespace LeftBound38911
def owner : Owner := ⟨.program ⟨214⟩, ⟨28982⟩⟩
def transferEvent : Nat := 38911
def frameStart : Nat := 38811
def rule : BoundRule := .sum [.predecessor 0 38909 .coefficient, .predecessor 1 38910 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38909 .coefficient)
      LeftBound38907.bound (LeftBound38907.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events151.exact38908RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38907.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38907.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38910 .coefficient)
      LeftBound38888.bound (LeftBound38888.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events151.exact38893RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38888.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38888.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound38907.bound, LeftBound38888.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38907.bound, LeftBound38888.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound38907.actual selector witness, LeftBound38888.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound38911

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
