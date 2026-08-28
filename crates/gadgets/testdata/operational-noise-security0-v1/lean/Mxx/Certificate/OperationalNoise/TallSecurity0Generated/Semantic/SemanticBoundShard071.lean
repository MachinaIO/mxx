import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard070

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound11760
def owner : Owner := ⟨.program ⟨214⟩, ⟨27920⟩⟩
def transferEvent : Nat := 11760
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 11758 .coefficient) (.predecessor 1 11759 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11758 .coefficient)
      LeftBound11753.bound (LeftBound11753.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events045.exact11757RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11753.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11753.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11759 .coefficient)
      LeftAuthority11460.bound (LeftAuthority11460.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11461RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11460.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11460.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound11753.bound LeftAuthority11460.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11753.bound, LeftAuthority11460.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound11753.actual selector witness) * (LeftAuthority11460.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound11760

namespace LeftBound11761
def owner : Owner := ⟨.program ⟨214⟩, ⟨27920⟩⟩
def transferEvent : Nat := 11761
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27918⟩⟩]⟩ [⟨.result 11461 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11461 .coefficient)
      LeftAuthority11460.bound (LeftAuthority11460.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27918⟩⟩) (rawTerms := some (Proof.Events044.exact11461RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11460.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11460.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority11460.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11460.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority11460.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound11761

namespace LeftBound11762
def owner : Owner := ⟨.program ⟨214⟩, ⟨27920⟩⟩
def transferEvent : Nat := 11762
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 11757 .summary) (.transfer 11761) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11757 .summary)
      LeftBound11756.bound (LeftBound11756.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26088⟩⟩) (rawTerms := some (Proof.Events045.exact11757RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound11756.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 11761)
      LeftBound11761.bound (LeftBound11761.actual selector witness) := by
  exact .transfer (LeftBound11761.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound11756.bound LeftBound11761.bound
def bound : CoeffClass := .finite ⟨1292068472128282820608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11756.bound, LeftBound11761.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound11756.actual selector witness) * (LeftBound11761.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound11762

namespace LeftBound11773
def owner : Owner := ⟨.program ⟨214⟩, ⟨21418⟩⟩
def transferEvent : Nat := 11773
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 11771 .coefficient) (.value (.predecessor 1 11772 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11771 .coefficient)
      LeftAuthority11769.bound (LeftAuthority11769.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events045.exact11770RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11769.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11769.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11772 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority11769.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11769.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority11769.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound11773

namespace LeftBound11777
def owner : Owner := ⟨.program ⟨214⟩, ⟨21419⟩⟩
def transferEvent : Nat := 11777
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 11775 .coefficient) (.predecessor 1 11776 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11775 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11776 .coefficient)
      LeftBound11773.bound (LeftBound11773.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events045.exact11774RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11773.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11773.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound11773.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound11773.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound11773.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound11777

namespace LeftBound11778
def owner : Owner := ⟨.program ⟨214⟩, ⟨21419⟩⟩
def transferEvent : Nat := 11778
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21416⟩⟩]⟩ [⟨.result 11770 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11770 .coefficient)
      LeftAuthority11769.bound (LeftAuthority11769.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21416⟩⟩) (rawTerms := some (Proof.Events045.exact11770RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11769.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11769.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority11769.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11769.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority11769.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound11778

namespace LeftBound11779
def owner : Owner := ⟨.program ⟨214⟩, ⟨21419⟩⟩
def transferEvent : Nat := 11779
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 11778) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 11778)
      LeftBound11778.bound (LeftBound11778.actual selector witness) := by
  exact .transfer (LeftBound11778.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound11778.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound11778.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound11778.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound11779

namespace LeftBound11874
def owner : Owner := ⟨.program ⟨214⟩, ⟨15957⟩⟩
def transferEvent : Nat := 11874
def frameStart : Nat := 11835
def rule : BoundRule := .identity (.predecessor 0 11873 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11873 .coefficient)
      LeftAuthority11871.bound (LeftAuthority11871.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11872RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11871.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11871.derived selector witness)

def rawBound : CoeffClass := LeftAuthority11871.bound
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11871.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority11871.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound11874

namespace LeftBound11891
def owner : Owner := ⟨.program ⟨214⟩, ⟨16031⟩⟩
def transferEvent : Nat := 11891
def frameStart : Nat := 11835
def rule : BoundRule := .sum [.predecessor 0 11889 .coefficient, .predecessor 1 11890 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11889 .coefficient)
      LeftBound11874.bound (LeftBound11874.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound11874.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11890 .coefficient)
      LeftAuthority11887.bound (LeftAuthority11887.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority11887.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11874.bound, LeftAuthority11887.bound]
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11874.bound, LeftAuthority11887.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound11874.actual selector witness, LeftAuthority11887.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11891

namespace LeftBound11894
def owner : Owner := ⟨.program ⟨214⟩, ⟨16032⟩⟩
def transferEvent : Nat := 11894
def frameStart : Nat := 11835
def rule : BoundRule := .identity (.predecessor 0 11893 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11893 .coefficient)
      LeftBound11891.bound (LeftBound11891.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound11891.derived selector witness)

def rawBound : CoeffClass := LeftBound11891.bound
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11891.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound11891.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound11894

namespace LeftBound11900
def owner : Owner := ⟨.program ⟨214⟩, ⟨16033⟩⟩
def transferEvent : Nat := 11900
def frameStart : Nat := 11835
def rule : BoundRule := .product (.predecessor 0 11898 .coefficient) (.predecessor 1 11899 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11898 .coefficient)
      LeftAuthority11896.bound (LeftAuthority11896.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11897RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11896.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11896.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11899 .coefficient)
      LeftBound11894.bound (LeftBound11894.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11895RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11894.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11894.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority11896.bound LeftBound11894.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11896.bound, LeftBound11894.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority11896.actual selector witness) * (LeftBound11894.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound11900

namespace LeftBound11908
def owner : Owner := ⟨.program ⟨214⟩, ⟨16034⟩⟩
def transferEvent : Nat := 11908
def frameStart : Nat := 11835
def rule : BoundRule := .sum [.predecessor 0 11906 .coefficient, .predecessor 1 11907 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11906 .coefficient)
      LeftAuthority11904.bound (LeftAuthority11904.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11905RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11904.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11904.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11907 .coefficient)
      LeftBound11900.bound (LeftBound11900.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11902RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11900.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11900.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority11904.bound, LeftBound11900.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11904.bound, LeftBound11900.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority11904.actual selector witness, LeftBound11900.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11908

namespace LeftBound11912
def owner : Owner := ⟨.program ⟨214⟩, ⟨27919⟩⟩
def transferEvent : Nat := 11912
def frameStart : Nat := 11835
def rule : BoundRule := .product (.predecessor 0 11910 .coefficient) (.predecessor 1 11911 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11910 .coefficient)
      LeftBound11908.bound (LeftBound11908.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11909RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11908.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11908.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11911 .coefficient)
      LeftAuthority11885.bound (LeftAuthority11885.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11886RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11885.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11885.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound11908.bound LeftAuthority11885.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11908.bound, LeftAuthority11885.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound11908.actual selector witness) * (LeftAuthority11885.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound11912

namespace LeftBound11923
def owner : Owner := ⟨.program ⟨214⟩, ⟨15999⟩⟩
def transferEvent : Nat := 11923
def frameStart : Nat := 11835
def rule : BoundRule := .product (.predecessor 0 11921 .coefficient) (.predecessor 1 11922 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11921 .coefficient)
      LeftAuthority11896.bound (LeftAuthority11896.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11897RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11896.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11896.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11922 .coefficient)
      LeftAuthority11919.bound (LeftAuthority11919.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11920RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11919.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11919.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority11896.bound LeftAuthority11919.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11896.bound, LeftAuthority11919.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority11896.actual selector witness) * (LeftAuthority11919.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound11923

namespace LeftBound11931
def owner : Owner := ⟨.program ⟨214⟩, ⟨16000⟩⟩
def transferEvent : Nat := 11931
def frameStart : Nat := 11835
def rule : BoundRule := .sum [.predecessor 0 11929 .coefficient, .predecessor 1 11930 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11929 .coefficient)
      LeftAuthority11927.bound (LeftAuthority11927.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11928RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11927.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11927.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11930 .coefficient)
      LeftBound11923.bound (LeftBound11923.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11925RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11923.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11923.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority11927.bound, LeftBound11923.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11927.bound, LeftBound11923.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority11927.actual selector witness, LeftBound11923.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11931

namespace LeftBound11935
def owner : Owner := ⟨.program ⟨214⟩, ⟨27923⟩⟩
def transferEvent : Nat := 11935
def frameStart : Nat := 11835
def rule : BoundRule := .sum [.predecessor 0 11933 .coefficient, .predecessor 1 11934 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11933 .coefficient)
      LeftBound11931.bound (LeftBound11931.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11932RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11931.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11931.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11934 .coefficient)
      LeftBound11912.bound (LeftBound11912.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11917RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11912.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11912.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11931.bound, LeftBound11912.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11931.bound, LeftBound11912.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound11931.actual selector witness, LeftBound11912.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11935

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
