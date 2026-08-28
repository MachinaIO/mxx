import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard664
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard718

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound104919
def owner : Owner := ⟨.program ⟨214⟩, ⟨28694⟩⟩
def transferEvent : Nat := 104919
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 97228 .summary) (.transfer 104918) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 97228 .summary)
      LeftBound97227.bound (LeftBound97227.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25208⟩⟩) (rawTerms := some (Proof.Events379.exact97228RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound97227.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 104918)
      LeftBound104918.bound (LeftBound104918.actual selector witness) := by
  exact .transfer (LeftBound104918.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound97227.bound LeftBound104918.bound
def bound : CoeffClass := .finite ⟨1292270184133468094464, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97227.bound, LeftBound104918.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound97227.actual selector witness) * (LeftBound104918.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104919

namespace LeftBound104930
def owner : Owner := ⟨.program ⟨214⟩, ⟨21895⟩⟩
def transferEvent : Nat := 104930
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 104928 .coefficient) (.value (.predecessor 1 104929 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104928 .coefficient)
      LeftAuthority104926.bound (LeftAuthority104926.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events409.exact104927RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104926.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104926.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104929 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority104926.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority104926.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority104926.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound104930

namespace LeftBound104934
def owner : Owner := ⟨.program ⟨214⟩, ⟨21896⟩⟩
def transferEvent : Nat := 104934
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 104932 .coefficient) (.predecessor 1 104933 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104932 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104933 .coefficient)
      LeftBound104930.bound (LeftBound104930.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events409.exact104931RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104930.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104930.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound104930.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound104930.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound104930.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104934

namespace LeftBound104935
def owner : Owner := ⟨.program ⟨214⟩, ⟨21896⟩⟩
def transferEvent : Nat := 104935
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21893⟩⟩]⟩ [⟨.result 104927 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 104927 .coefficient)
      LeftAuthority104926.bound (LeftAuthority104926.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21893⟩⟩) (rawTerms := some (Proof.Events409.exact104927RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104926.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104926.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority104926.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority104926.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority104926.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound104935

namespace LeftBound104936
def owner : Owner := ⟨.program ⟨214⟩, ⟨21896⟩⟩
def transferEvent : Nat := 104936
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 104935) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 104935)
      LeftBound104935.bound (LeftBound104935.actual selector witness) := by
  exact .transfer (LeftBound104935.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound104935.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound104935.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound104935.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104936

namespace LeftBound105007
def owner : Owner := ⟨.program ⟨214⟩, ⟨16372⟩⟩
def transferEvent : Nat := 105007
def frameStart : Nat := 104980
def rule : BoundRule := .identity (.predecessor 0 105006 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105006 .coefficient)
      LeftAuthority105004.bound (LeftAuthority105004.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events410.exact105005RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105004.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105004.derived selector witness)

def rawBound : CoeffClass := LeftAuthority105004.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105004.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority105004.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound105007

namespace LeftBound105024
def owner : Owner := ⟨.program ⟨214⟩, ⟨16413⟩⟩
def transferEvent : Nat := 105024
def frameStart : Nat := 104980
def rule : BoundRule := .sum [.predecessor 0 105022 .coefficient, .predecessor 1 105023 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105022 .coefficient)
      LeftBound105007.bound (LeftBound105007.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound105007.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105023 .coefficient)
      LeftAuthority105020.bound (LeftAuthority105020.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority105020.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound105007.bound, LeftAuthority105020.bound]
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105007.bound, LeftAuthority105020.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound105007.actual selector witness, LeftAuthority105020.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound105024

namespace LeftBound105027
def owner : Owner := ⟨.program ⟨214⟩, ⟨16414⟩⟩
def transferEvent : Nat := 105027
def frameStart : Nat := 104980
def rule : BoundRule := .identity (.predecessor 0 105026 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105026 .coefficient)
      LeftBound105024.bound (LeftBound105024.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound105024.derived selector witness)

def rawBound : CoeffClass := LeftBound105024.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105024.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound105024.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound105027

namespace LeftBound105033
def owner : Owner := ⟨.program ⟨214⟩, ⟨16415⟩⟩
def transferEvent : Nat := 105033
def frameStart : Nat := 104980
def rule : BoundRule := .product (.predecessor 0 105031 .coefficient) (.predecessor 1 105032 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105031 .coefficient)
      LeftAuthority105029.bound (LeftAuthority105029.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events410.exact105030RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105029.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105029.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105032 .coefficient)
      LeftBound105027.bound (LeftBound105027.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events410.exact105028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105027.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105027.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority105029.bound LeftBound105027.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105029.bound, LeftBound105027.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority105029.actual selector witness) * (LeftBound105027.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105033

namespace LeftBound105041
def owner : Owner := ⟨.program ⟨214⟩, ⟨16416⟩⟩
def transferEvent : Nat := 105041
def frameStart : Nat := 104980
def rule : BoundRule := .sum [.predecessor 0 105039 .coefficient, .predecessor 1 105040 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105039 .coefficient)
      LeftAuthority105037.bound (LeftAuthority105037.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events410.exact105038RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105037.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105037.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105040 .coefficient)
      LeftBound105033.bound (LeftBound105033.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events410.exact105035RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105033.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105033.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority105037.bound, LeftBound105033.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105037.bound, LeftBound105033.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority105037.actual selector witness, LeftBound105033.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound105041

namespace LeftBound105045
def owner : Owner := ⟨.program ⟨214⟩, ⟨28693⟩⟩
def transferEvent : Nat := 105045
def frameStart : Nat := 104980
def rule : BoundRule := .product (.predecessor 0 105043 .coefficient) (.predecessor 1 105044 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105043 .coefficient)
      LeftBound105041.bound (LeftBound105041.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events410.exact105042RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105041.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105041.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105044 .coefficient)
      LeftAuthority105018.bound (LeftAuthority105018.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events410.exact105019RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105018.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105018.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound105041.bound LeftAuthority105018.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105041.bound, LeftAuthority105018.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound105041.actual selector witness) * (LeftAuthority105018.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105045

namespace LeftBound105056
def owner : Owner := ⟨.program ⟨214⟩, ⟨18801⟩⟩
def transferEvent : Nat := 105056
def frameStart : Nat := 104980
def rule : BoundRule := .product (.predecessor 0 105054 .coefficient) (.predecessor 1 105055 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105054 .coefficient)
      LeftAuthority105029.bound (LeftAuthority105029.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events410.exact105030RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105029.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105029.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105055 .coefficient)
      LeftAuthority105052.bound (LeftAuthority105052.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events410.exact105053RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105052.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105052.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority105029.bound LeftAuthority105052.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105029.bound, LeftAuthority105052.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority105029.actual selector witness) * (LeftAuthority105052.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105056

namespace LeftBound105064
def owner : Owner := ⟨.program ⟨214⟩, ⟨18805⟩⟩
def transferEvent : Nat := 105064
def frameStart : Nat := 104980
def rule : BoundRule := .sum [.predecessor 0 105062 .coefficient, .predecessor 1 105063 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105062 .coefficient)
      LeftAuthority105060.bound (LeftAuthority105060.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events410.exact105061RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105060.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105060.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105063 .coefficient)
      LeftBound105056.bound (LeftBound105056.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events410.exact105058RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105056.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105056.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority105060.bound, LeftBound105056.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105060.bound, LeftBound105056.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority105060.actual selector witness, LeftBound105056.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound105064

namespace LeftBound105068
def owner : Owner := ⟨.program ⟨214⟩, ⟨28698⟩⟩
def transferEvent : Nat := 105068
def frameStart : Nat := 104980
def rule : BoundRule := .sum [.predecessor 0 105066 .coefficient, .predecessor 1 105067 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105066 .coefficient)
      LeftBound105064.bound (LeftBound105064.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events410.exact105065RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105064.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105064.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105067 .coefficient)
      LeftBound105045.bound (LeftBound105045.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events410.exact105050RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105045.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105045.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound105064.bound, LeftBound105045.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105064.bound, LeftBound105045.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound105064.actual selector witness, LeftBound105045.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound105068

namespace LeftBound105081
def owner : Owner := ⟨.program ⟨214⟩, ⟨28695⟩⟩
def transferEvent : Nat := 105081
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 105079 .coefficient, .predecessor 1 105080 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105079 .coefficient)
      LeftBound104934.bound (LeftBound104934.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events410.exact105078RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104934.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104934.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105080 .coefficient)
      LeftBound104917.bound (LeftBound104917.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events409.exact104924RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104917.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104917.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104934.bound, LeftBound104917.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104934.bound, LeftBound104917.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound104934.actual selector witness, LeftBound104917.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound105081

namespace LeftBound105084
def owner : Owner := ⟨.program ⟨214⟩, ⟨28695⟩⟩
def transferEvent : Nat := 105084
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 105078 .summary, .result 104924 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 105078 .summary)
      LeftBound104936.bound (LeftBound104936.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21896⟩⟩) (rawTerms := some (Proof.Events410.exact105078RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104936.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 104924 .summary)
      LeftBound104919.bound (LeftBound104919.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28694⟩⟩) (rawTerms := some (Proof.Events409.exact104924RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104919.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104936.bound, LeftBound104919.bound]
def bound : CoeffClass := .finite ⟨1292270185944771604480, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104936.bound, LeftBound104919.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound104936.actual selector witness, LeftBound104919.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound105084

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
