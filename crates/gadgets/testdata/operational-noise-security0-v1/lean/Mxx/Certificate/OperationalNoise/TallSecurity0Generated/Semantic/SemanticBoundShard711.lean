import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard642

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound103789
def owner : Owner := ⟨.program ⟨214⟩, ⟨30056⟩⟩
def transferEvent : Nat := 103789
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 103787 .coefficient) (.predecessor 1 103788 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103787 .coefficient)
      LeftBound94620.bound (LeftBound94620.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events369.exact94624RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94620.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94620.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103788 .coefficient)
      LeftAuthority103785.bound (LeftAuthority103785.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events405.exact103786RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103785.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103785.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound94620.bound LeftAuthority103785.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94620.bound, LeftAuthority103785.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound94620.actual selector witness) * (LeftAuthority103785.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound103789

namespace LeftBound103790
def owner : Owner := ⟨.program ⟨214⟩, ⟨30056⟩⟩
def transferEvent : Nat := 103790
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨30054⟩⟩]⟩ [⟨.result 103786 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 103786 .coefficient)
      LeftAuthority103785.bound (LeftAuthority103785.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨30054⟩⟩) (rawTerms := some (Proof.Events405.exact103786RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103785.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103785.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority103785.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority103785.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority103785.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound103790

namespace LeftBound103791
def owner : Owner := ⟨.program ⟨214⟩, ⟨30056⟩⟩
def transferEvent : Nat := 103791
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94624 .summary) (.transfer 103790) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94624 .summary)
      LeftBound94623.bound (LeftBound94623.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25747⟩⟩) (rawTerms := some (Proof.Events369.exact94624RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94623.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 103790)
      LeftBound103790.bound (LeftBound103790.actual selector witness) := by
  exact .transfer (LeftBound103790.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound94623.bound LeftBound103790.bound
def bound : CoeffClass := .finite ⟨1292539133473715126272, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94623.bound, LeftBound103790.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound94623.actual selector witness) * (LeftBound103790.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound103791

namespace LeftBound103802
def owner : Owner := ⟨.program ⟨214⟩, ⟨22759⟩⟩
def transferEvent : Nat := 103802
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 103800 .coefficient) (.value (.predecessor 1 103801 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103800 .coefficient)
      LeftAuthority103798.bound (LeftAuthority103798.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events405.exact103799RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103798.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103798.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103801 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority103798.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority103798.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority103798.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound103802

namespace LeftBound103806
def owner : Owner := ⟨.program ⟨214⟩, ⟨22760⟩⟩
def transferEvent : Nat := 103806
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 103804 .coefficient) (.predecessor 1 103805 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103804 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103805 .coefficient)
      LeftBound103802.bound (LeftBound103802.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events405.exact103803RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103802.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103802.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound103802.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound103802.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound103802.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound103806

namespace LeftBound103807
def owner : Owner := ⟨.program ⟨214⟩, ⟨22760⟩⟩
def transferEvent : Nat := 103807
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22757⟩⟩]⟩ [⟨.result 103799 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 103799 .coefficient)
      LeftAuthority103798.bound (LeftAuthority103798.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22757⟩⟩) (rawTerms := some (Proof.Events405.exact103799RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103798.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103798.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority103798.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority103798.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority103798.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound103807

namespace LeftBound103808
def owner : Owner := ⟨.program ⟨214⟩, ⟨22760⟩⟩
def transferEvent : Nat := 103808
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 103807) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 103807)
      LeftBound103807.bound (LeftBound103807.actual selector witness) := by
  exact .transfer (LeftBound103807.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound103807.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound103807.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound103807.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound103808

namespace LeftBound103879
def owner : Owner := ⟨.program ⟨214⟩, ⟨17002⟩⟩
def transferEvent : Nat := 103879
def frameStart : Nat := 103852
def rule : BoundRule := .identity (.predecessor 0 103878 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103878 .coefficient)
      LeftAuthority103876.bound (LeftAuthority103876.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events405.exact103877RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103876.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103876.derived selector witness)

def rawBound : CoeffClass := LeftAuthority103876.bound
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority103876.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority103876.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound103879

namespace LeftBound103896
def owner : Owner := ⟨.program ⟨214⟩, ⟨17043⟩⟩
def transferEvent : Nat := 103896
def frameStart : Nat := 103852
def rule : BoundRule := .sum [.predecessor 0 103894 .coefficient, .predecessor 1 103895 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103894 .coefficient)
      LeftBound103879.bound (LeftBound103879.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound103879.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103895 .coefficient)
      LeftAuthority103892.bound (LeftAuthority103892.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority103892.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound103879.bound, LeftAuthority103892.bound]
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103879.bound, LeftAuthority103892.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound103879.actual selector witness, LeftAuthority103892.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103896

namespace LeftBound103899
def owner : Owner := ⟨.program ⟨214⟩, ⟨17044⟩⟩
def transferEvent : Nat := 103899
def frameStart : Nat := 103852
def rule : BoundRule := .identity (.predecessor 0 103898 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103898 .coefficient)
      LeftBound103896.bound (LeftBound103896.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound103896.derived selector witness)

def rawBound : CoeffClass := LeftBound103896.bound
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103896.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound103896.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound103899

namespace LeftBound103905
def owner : Owner := ⟨.program ⟨214⟩, ⟨17045⟩⟩
def transferEvent : Nat := 103905
def frameStart : Nat := 103852
def rule : BoundRule := .product (.predecessor 0 103903 .coefficient) (.predecessor 1 103904 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103903 .coefficient)
      LeftAuthority103901.bound (LeftAuthority103901.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events405.exact103902RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103901.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103901.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103904 .coefficient)
      LeftBound103899.bound (LeftBound103899.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events405.exact103900RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103899.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103899.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority103901.bound LeftBound103899.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority103901.bound, LeftBound103899.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority103901.actual selector witness) * (LeftBound103899.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound103905

namespace LeftBound103913
def owner : Owner := ⟨.program ⟨214⟩, ⟨17046⟩⟩
def transferEvent : Nat := 103913
def frameStart : Nat := 103852
def rule : BoundRule := .sum [.predecessor 0 103911 .coefficient, .predecessor 1 103912 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103911 .coefficient)
      LeftAuthority103909.bound (LeftAuthority103909.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events405.exact103910RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103909.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103909.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103912 .coefficient)
      LeftBound103905.bound (LeftBound103905.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events405.exact103907RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103905.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103905.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority103909.bound, LeftBound103905.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority103909.bound, LeftBound103905.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority103909.actual selector witness, LeftBound103905.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103913

namespace LeftBound103917
def owner : Owner := ⟨.program ⟨214⟩, ⟨30055⟩⟩
def transferEvent : Nat := 103917
def frameStart : Nat := 103852
def rule : BoundRule := .product (.predecessor 0 103915 .coefficient) (.predecessor 1 103916 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103915 .coefficient)
      LeftBound103913.bound (LeftBound103913.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events405.exact103914RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103913.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103913.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103916 .coefficient)
      LeftAuthority103890.bound (LeftAuthority103890.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events405.exact103891RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103890.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103890.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound103913.bound LeftAuthority103890.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103913.bound, LeftAuthority103890.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound103913.actual selector witness) * (LeftAuthority103890.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound103917

namespace LeftBound103928
def owner : Owner := ⟨.program ⟨214⟩, ⟨18116⟩⟩
def transferEvent : Nat := 103928
def frameStart : Nat := 103852
def rule : BoundRule := .product (.predecessor 0 103926 .coefficient) (.predecessor 1 103927 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103926 .coefficient)
      LeftAuthority103901.bound (LeftAuthority103901.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events405.exact103902RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103901.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103901.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103927 .coefficient)
      LeftAuthority103924.bound (LeftAuthority103924.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events405.exact103925RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103924.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103924.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority103901.bound LeftAuthority103924.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority103901.bound, LeftAuthority103924.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority103901.actual selector witness) * (LeftAuthority103924.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound103928

namespace LeftBound103936
def owner : Owner := ⟨.program ⟨214⟩, ⟨18117⟩⟩
def transferEvent : Nat := 103936
def frameStart : Nat := 103852
def rule : BoundRule := .sum [.predecessor 0 103934 .coefficient, .predecessor 1 103935 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103934 .coefficient)
      LeftAuthority103932.bound (LeftAuthority103932.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events405.exact103933RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103932.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103932.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103935 .coefficient)
      LeftBound103928.bound (LeftBound103928.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events405.exact103930RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103928.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103928.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority103932.bound, LeftBound103928.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority103932.bound, LeftBound103928.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority103932.actual selector witness, LeftBound103928.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103936

namespace LeftBound103940
def owner : Owner := ⟨.program ⟨214⟩, ⟨30060⟩⟩
def transferEvent : Nat := 103940
def frameStart : Nat := 103852
def rule : BoundRule := .sum [.predecessor 0 103938 .coefficient, .predecessor 1 103939 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103938 .coefficient)
      LeftBound103936.bound (LeftBound103936.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events406.exact103937RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103936.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103936.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103939 .coefficient)
      LeftBound103917.bound (LeftBound103917.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events405.exact103922RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103917.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103917.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound103936.bound, LeftBound103917.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103936.bound, LeftBound103917.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound103936.actual selector witness, LeftBound103917.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103940

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
