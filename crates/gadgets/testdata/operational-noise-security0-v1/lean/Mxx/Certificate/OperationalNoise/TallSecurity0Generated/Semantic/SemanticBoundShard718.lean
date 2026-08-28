import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard019
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard664
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard717

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound104819
def owner : Owner := ⟨.program ⟨214⟩, ⟨16456⟩⟩
def transferEvent : Nat := 104819
def frameStart : Nat := 104792
def rule : BoundRule := .identity (.predecessor 0 104818 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104818 .coefficient)
      LeftAuthority104816.bound (LeftAuthority104816.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events409.exact104817RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104816.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104816.derived selector witness)

def rawBound : CoeffClass := LeftAuthority104816.bound
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority104816.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority104816.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound104819

namespace LeftBound104836
def owner : Owner := ⟨.program ⟨214⟩, ⟨16497⟩⟩
def transferEvent : Nat := 104836
def frameStart : Nat := 104792
def rule : BoundRule := .sum [.predecessor 0 104834 .coefficient, .predecessor 1 104835 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104834 .coefficient)
      LeftBound104819.bound (LeftBound104819.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound104819.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104835 .coefficient)
      LeftAuthority104832.bound (LeftAuthority104832.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority104832.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104819.bound, LeftAuthority104832.bound]
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104819.bound, LeftAuthority104832.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound104819.actual selector witness, LeftAuthority104832.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104836

namespace LeftBound104839
def owner : Owner := ⟨.program ⟨214⟩, ⟨16498⟩⟩
def transferEvent : Nat := 104839
def frameStart : Nat := 104792
def rule : BoundRule := .identity (.predecessor 0 104838 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104838 .coefficient)
      LeftBound104836.bound (LeftBound104836.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound104836.derived selector witness)

def rawBound : CoeffClass := LeftBound104836.bound
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104836.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound104836.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound104839

namespace LeftBound104845
def owner : Owner := ⟨.program ⟨214⟩, ⟨16499⟩⟩
def transferEvent : Nat := 104845
def frameStart : Nat := 104792
def rule : BoundRule := .product (.predecessor 0 104843 .coefficient) (.predecessor 1 104844 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104843 .coefficient)
      LeftAuthority104841.bound (LeftAuthority104841.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events409.exact104842RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104841.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104841.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104844 .coefficient)
      LeftBound104839.bound (LeftBound104839.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events409.exact104840RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104839.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104839.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority104841.bound LeftBound104839.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority104841.bound, LeftBound104839.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority104841.actual selector witness) * (LeftBound104839.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104845

namespace LeftBound104853
def owner : Owner := ⟨.program ⟨214⟩, ⟨16500⟩⟩
def transferEvent : Nat := 104853
def frameStart : Nat := 104792
def rule : BoundRule := .sum [.predecessor 0 104851 .coefficient, .predecessor 1 104852 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104851 .coefficient)
      LeftAuthority104849.bound (LeftAuthority104849.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events409.exact104850RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104849.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104849.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104852 .coefficient)
      LeftBound104845.bound (LeftBound104845.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events409.exact104847RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104845.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104845.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority104849.bound, LeftBound104845.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority104849.bound, LeftBound104845.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority104849.actual selector witness, LeftBound104845.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104853

namespace LeftBound104857
def owner : Owner := ⟨.program ⟨214⟩, ⟨28910⟩⟩
def transferEvent : Nat := 104857
def frameStart : Nat := 104792
def rule : BoundRule := .product (.predecessor 0 104855 .coefficient) (.predecessor 1 104856 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104855 .coefficient)
      LeftBound104853.bound (LeftBound104853.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events409.exact104854RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104853.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104853.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104856 .coefficient)
      LeftAuthority104830.bound (LeftAuthority104830.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events409.exact104831RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104830.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104830.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound104853.bound LeftAuthority104830.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104853.bound, LeftAuthority104830.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound104853.actual selector witness) * (LeftAuthority104830.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104857

namespace LeftBound104868
def owner : Owner := ⟨.program ⟨214⟩, ⟨17542⟩⟩
def transferEvent : Nat := 104868
def frameStart : Nat := 104792
def rule : BoundRule := .product (.predecessor 0 104866 .coefficient) (.predecessor 1 104867 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104866 .coefficient)
      LeftAuthority104841.bound (LeftAuthority104841.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events409.exact104842RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104841.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104841.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104867 .coefficient)
      LeftAuthority104864.bound (LeftAuthority104864.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events409.exact104865RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104864.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104864.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority104841.bound LeftAuthority104864.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority104841.bound, LeftAuthority104864.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority104841.actual selector witness) * (LeftAuthority104864.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104868

namespace LeftBound104876
def owner : Owner := ⟨.program ⟨214⟩, ⟨17543⟩⟩
def transferEvent : Nat := 104876
def frameStart : Nat := 104792
def rule : BoundRule := .sum [.predecessor 0 104874 .coefficient, .predecessor 1 104875 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104874 .coefficient)
      LeftAuthority104872.bound (LeftAuthority104872.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events409.exact104873RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104872.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104872.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104875 .coefficient)
      LeftBound104868.bound (LeftBound104868.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events409.exact104870RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104868.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104868.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority104872.bound, LeftBound104868.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority104872.bound, LeftBound104868.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority104872.actual selector witness, LeftBound104868.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104876

namespace LeftBound104880
def owner : Owner := ⟨.program ⟨214⟩, ⟨28915⟩⟩
def transferEvent : Nat := 104880
def frameStart : Nat := 104792
def rule : BoundRule := .sum [.predecessor 0 104878 .coefficient, .predecessor 1 104879 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104878 .coefficient)
      LeftBound104876.bound (LeftBound104876.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events409.exact104877RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104876.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104876.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104879 .coefficient)
      LeftBound104857.bound (LeftBound104857.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events409.exact104862RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104857.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104857.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104876.bound, LeftBound104857.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104876.bound, LeftBound104857.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound104876.actual selector witness, LeftBound104857.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104880

namespace LeftBound104893
def owner : Owner := ⟨.program ⟨214⟩, ⟨28912⟩⟩
def transferEvent : Nat := 104893
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 104891 .coefficient, .predecessor 1 104892 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104891 .coefficient)
      LeftBound104746.bound (LeftBound104746.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events409.exact104890RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104746.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104746.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104892 .coefficient)
      LeftBound104729.bound (LeftBound104729.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events409.exact104736RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104729.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104729.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104746.bound, LeftBound104729.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104746.bound, LeftBound104729.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound104746.actual selector witness, LeftBound104729.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104893

namespace LeftBound104896
def owner : Owner := ⟨.program ⟨214⟩, ⟨28912⟩⟩
def transferEvent : Nat := 104896
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 104890 .summary, .result 104736 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 104890 .summary)
      LeftBound104748.bound (LeftBound104748.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22040⟩⟩) (rawTerms := some (Proof.Events409.exact104890RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104748.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 104736 .summary)
      LeftBound104731.bound (LeftBound104731.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28911⟩⟩) (rawTerms := some (Proof.Events409.exact104736RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104731.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104748.bound, LeftBound104731.bound]
def bound : CoeffClass := .finite ⟨1292315010834812776448, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104748.bound, LeftBound104731.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound104748.actual selector witness, LeftBound104731.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104896

namespace LeftBound104900
def owner : Owner := ⟨.program ⟨214⟩, ⟨28913⟩⟩
def transferEvent : Nat := 104900
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 104898 .coefficient) (.predecessor 1 104899 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104898 .coefficient)
      LeftBound104893.bound (LeftBound104893.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events409.exact104897RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104893.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104893.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104899 .coefficient)
      LeftBound5618.bound (LeftBound5618.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5619RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5618.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5618.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound104893.bound LeftBound5618.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104893.bound, LeftBound5618.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound104893.actual selector witness) * (LeftBound5618.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104900

namespace LeftBound104901
def owner : Owner := ⟨.program ⟨214⟩, ⟨28913⟩⟩
def transferEvent : Nat := 104901
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6669⟩⟩]⟩ [⟨.result 5615 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5615 .coefficient)
      LeftAuthority5614.bound (LeftAuthority5614.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6669⟩⟩) (rawTerms := some (Proof.Events021.exact5615RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5614.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5614.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5614.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5614.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5614.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound104901

namespace LeftBound104902
def owner : Owner := ⟨.program ⟨214⟩, ⟨28913⟩⟩
def transferEvent : Nat := 104902
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 104897 .summary) (.transfer 104901) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 104897 .summary)
      LeftBound104896.bound (LeftBound104896.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28912⟩⟩) (rawTerms := some (Proof.Events409.exact104897RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104896.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 104901)
      LeftBound104901.bound (LeftBound104901.actual selector witness) := by
  exact .transfer (LeftBound104901.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound104896.bound LeftBound104901.bound
def bound : CoeffClass := .finite ⟨4742816766803936246568583168, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104896.bound, LeftBound104901.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound104896.actual selector witness) * (LeftBound104901.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104902

namespace LeftBound104917
def owner : Owner := ⟨.program ⟨214⟩, ⟨28694⟩⟩
def transferEvent : Nat := 104917
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 104915 .coefficient) (.predecessor 1 104916 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104915 .coefficient)
      LeftBound97224.bound (LeftBound97224.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events379.exact97228RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97224.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97224.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104916 .coefficient)
      LeftAuthority104913.bound (LeftAuthority104913.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events409.exact104914RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104913.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104913.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound97224.bound LeftAuthority104913.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97224.bound, LeftAuthority104913.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound97224.actual selector witness) * (LeftAuthority104913.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104917

namespace LeftBound104918
def owner : Owner := ⟨.program ⟨214⟩, ⟨28694⟩⟩
def transferEvent : Nat := 104918
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28692⟩⟩]⟩ [⟨.result 104914 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 104914 .coefficient)
      LeftAuthority104913.bound (LeftAuthority104913.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28692⟩⟩) (rawTerms := some (Proof.Events409.exact104914RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104913.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104913.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority104913.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority104913.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority104913.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound104918

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
