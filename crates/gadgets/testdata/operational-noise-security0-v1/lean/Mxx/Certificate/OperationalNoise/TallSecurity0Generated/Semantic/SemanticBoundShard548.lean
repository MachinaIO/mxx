import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard541
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard547

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound80692
def owner : Owner := ⟨.program ⟨214⟩, ⟨22698⟩⟩
def transferEvent : Nat := 80692
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 80690 .coefficient) (.value (.predecessor 1 80691 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80690 .coefficient)
      LeftAuthority80688.bound (LeftAuthority80688.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events315.exact80689RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80688.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80688.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80691 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority80688.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority80688.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority80688.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound80692

namespace LeftBound80696
def owner : Owner := ⟨.program ⟨214⟩, ⟨22699⟩⟩
def transferEvent : Nat := 80696
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 80694 .coefficient) (.predecessor 1 80695 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80694 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80695 .coefficient)
      LeftBound80692.bound (LeftBound80692.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events315.exact80693RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80692.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80692.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound80692.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound80692.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound80692.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound80696

namespace LeftBound80697
def owner : Owner := ⟨.program ⟨214⟩, ⟨22699⟩⟩
def transferEvent : Nat := 80697
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22696⟩⟩]⟩ [⟨.result 80689 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80689 .coefficient)
      LeftAuthority80688.bound (LeftAuthority80688.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22696⟩⟩) (rawTerms := some (Proof.Events315.exact80689RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80688.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80688.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority80688.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority80688.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority80688.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound80697

namespace LeftBound80698
def owner : Owner := ⟨.program ⟨214⟩, ⟨22699⟩⟩
def transferEvent : Nat := 80698
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 80697) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 80697)
      LeftBound80697.bound (LeftBound80697.actual selector witness) := by
  exact .transfer (LeftBound80697.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound80697.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound80697.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound80697.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound80698

namespace LeftBound80793
def owner : Owner := ⟨.program ⟨214⟩, ⟨16872⟩⟩
def transferEvent : Nat := 80793
def frameStart : Nat := 80754
def rule : BoundRule := .identity (.predecessor 0 80792 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80792 .coefficient)
      LeftAuthority80790.bound (LeftAuthority80790.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events315.exact80791RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80790.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80790.derived selector witness)

def rawBound : CoeffClass := LeftAuthority80790.bound
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority80790.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority80790.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound80793

namespace LeftBound80810
def owner : Owner := ⟨.program ⟨214⟩, ⟨16967⟩⟩
def transferEvent : Nat := 80810
def frameStart : Nat := 80754
def rule : BoundRule := .sum [.predecessor 0 80808 .coefficient, .predecessor 1 80809 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80808 .coefficient)
      LeftBound80793.bound (LeftBound80793.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound80793.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80809 .coefficient)
      LeftAuthority80806.bound (LeftAuthority80806.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority80806.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound80793.bound, LeftAuthority80806.bound]
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80793.bound, LeftAuthority80806.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound80793.actual selector witness, LeftAuthority80806.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound80810

namespace LeftBound80813
def owner : Owner := ⟨.program ⟨214⟩, ⟨16968⟩⟩
def transferEvent : Nat := 80813
def frameStart : Nat := 80754
def rule : BoundRule := .identity (.predecessor 0 80812 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80812 .coefficient)
      LeftBound80810.bound (LeftBound80810.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound80810.derived selector witness)

def rawBound : CoeffClass := LeftBound80810.bound
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80810.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound80810.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound80813

namespace LeftBound80819
def owner : Owner := ⟨.program ⟨214⟩, ⟨16969⟩⟩
def transferEvent : Nat := 80819
def frameStart : Nat := 80754
def rule : BoundRule := .product (.predecessor 0 80817 .coefficient) (.predecessor 1 80818 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80817 .coefficient)
      LeftAuthority80815.bound (LeftAuthority80815.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events315.exact80816RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80815.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80815.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80818 .coefficient)
      LeftBound80813.bound (LeftBound80813.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events315.exact80814RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80813.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80813.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority80815.bound LeftBound80813.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority80815.bound, LeftBound80813.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority80815.actual selector witness) * (LeftBound80813.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound80819

namespace LeftBound80827
def owner : Owner := ⟨.program ⟨214⟩, ⟨16970⟩⟩
def transferEvent : Nat := 80827
def frameStart : Nat := 80754
def rule : BoundRule := .sum [.predecessor 0 80825 .coefficient, .predecessor 1 80826 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80825 .coefficient)
      LeftAuthority80823.bound (LeftAuthority80823.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events315.exact80824RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80823.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80823.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80826 .coefficient)
      LeftBound80819.bound (LeftBound80819.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events315.exact80821RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80819.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80819.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority80823.bound, LeftBound80819.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority80823.bound, LeftBound80819.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority80823.actual selector witness, LeftBound80819.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound80827

namespace LeftBound80831
def owner : Owner := ⟨.program ⟨214⟩, ⟨29820⟩⟩
def transferEvent : Nat := 80831
def frameStart : Nat := 80754
def rule : BoundRule := .product (.predecessor 0 80829 .coefficient) (.predecessor 1 80830 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80829 .coefficient)
      LeftBound80827.bound (LeftBound80827.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events315.exact80828RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80827.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80827.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80830 .coefficient)
      LeftAuthority80804.bound (LeftAuthority80804.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events315.exact80805RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80804.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80804.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound80827.bound LeftAuthority80804.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80827.bound, LeftAuthority80804.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound80827.actual selector witness) * (LeftAuthority80804.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound80831

namespace LeftBound80842
def owner : Owner := ⟨.program ⟨214⟩, ⟨17086⟩⟩
def transferEvent : Nat := 80842
def frameStart : Nat := 80754
def rule : BoundRule := .product (.predecessor 0 80840 .coefficient) (.predecessor 1 80841 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80840 .coefficient)
      LeftAuthority80815.bound (LeftAuthority80815.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events315.exact80816RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80815.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80815.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80841 .coefficient)
      LeftAuthority80838.bound (LeftAuthority80838.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events315.exact80839RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80838.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80838.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority80815.bound LeftAuthority80838.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority80815.bound, LeftAuthority80838.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority80815.actual selector witness) * (LeftAuthority80838.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound80842

namespace LeftBound80850
def owner : Owner := ⟨.program ⟨214⟩, ⟨17087⟩⟩
def transferEvent : Nat := 80850
def frameStart : Nat := 80754
def rule : BoundRule := .sum [.predecessor 0 80848 .coefficient, .predecessor 1 80849 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80848 .coefficient)
      LeftAuthority80846.bound (LeftAuthority80846.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events315.exact80847RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80846.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80846.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80849 .coefficient)
      LeftBound80842.bound (LeftBound80842.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events315.exact80844RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80842.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80842.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority80846.bound, LeftBound80842.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority80846.bound, LeftBound80842.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority80846.actual selector witness, LeftBound80842.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound80850

namespace LeftBound80854
def owner : Owner := ⟨.program ⟨214⟩, ⟨29824⟩⟩
def transferEvent : Nat := 80854
def frameStart : Nat := 80754
def rule : BoundRule := .sum [.predecessor 0 80852 .coefficient, .predecessor 1 80853 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80852 .coefficient)
      LeftBound80850.bound (LeftBound80850.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events315.exact80851RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80850.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80850.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80853 .coefficient)
      LeftBound80831.bound (LeftBound80831.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events315.exact80836RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80831.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80831.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound80850.bound, LeftBound80831.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80850.bound, LeftBound80831.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound80850.actual selector witness, LeftBound80831.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound80854

namespace LeftBound80867
def owner : Owner := ⟨.program ⟨214⟩, ⟨29822⟩⟩
def transferEvent : Nat := 80867
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 80865 .coefficient, .predecessor 1 80866 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80865 .coefficient)
      LeftBound80696.bound (LeftBound80696.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events315.exact80864RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80696.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80696.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80866 .coefficient)
      LeftBound80679.bound (LeftBound80679.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events315.exact80686RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80679.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80679.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound80696.bound, LeftBound80679.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80696.bound, LeftBound80679.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound80696.actual selector witness, LeftBound80679.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound80867

namespace LeftBound80870
def owner : Owner := ⟨.program ⟨214⟩, ⟨29822⟩⟩
def transferEvent : Nat := 80870
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 80864 .summary, .result 80686 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80864 .summary)
      LeftBound80698.bound (LeftBound80698.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22699⟩⟩) (rawTerms := some (Proof.Events315.exact80864RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80698.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80686 .summary)
      LeftBound80681.bound (LeftBound80681.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29821⟩⟩) (rawTerms := some (Proof.Events315.exact80686RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80681.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound80698.bound, LeftBound80681.bound]
def bound : CoeffClass := .finite ⟨1292516722839998050304, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80698.bound, LeftBound80681.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound80698.actual selector witness, LeftBound80681.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound80870

namespace LeftBound80894
def owner : Owner := ⟨.program ⟨214⟩, ⟨12961⟩⟩
def transferEvent : Nat := 80894
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 80892 .coefficient) (.predecessor 1 80893 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80892 .coefficient)
      LeftAuthority3873.bound (LeftAuthority3873.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events015.exact3874RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3873.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3873.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80893 .coefficient)
      LeftBound79918.bound (LeftBound79918.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79920RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79918.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79918.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority3873.bound LeftBound79918.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3873.bound, LeftBound79918.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority3873.actual selector witness) * (LeftBound79918.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound80894

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
