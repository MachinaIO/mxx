import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard044
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard234
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard237
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard250

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound37789
def owner : Owner := ⟨.program ⟨214⟩, ⟨22419⟩⟩
def transferEvent : Nat := 37789
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 37787 .coefficient) (.predecessor 1 37788 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37787 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37788 .coefficient)
      LeftBound37785.bound (LeftBound37785.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events147.exact37786RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37785.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37785.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound37785.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound37785.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound37785.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound37789

namespace LeftBound37790
def owner : Owner := ⟨.program ⟨214⟩, ⟨22419⟩⟩
def transferEvent : Nat := 37790
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22416⟩⟩]⟩ [⟨.result 37782 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 37782 .coefficient)
      LeftAuthority37781.bound (LeftAuthority37781.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22416⟩⟩) (rawTerms := some (Proof.Events147.exact37782RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37781.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37781.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority37781.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority37781.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority37781.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound37790

namespace LeftBound37791
def owner : Owner := ⟨.program ⟨214⟩, ⟨22419⟩⟩
def transferEvent : Nat := 37791
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 37790) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 37790)
      LeftBound37790.bound (LeftBound37790.actual selector witness) := by
  exact .transfer (LeftBound37790.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound37790.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound37790.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound37790.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound37791

namespace LeftBound37886
def owner : Owner := ⟨.program ⟨214⟩, ⟨16642⟩⟩
def transferEvent : Nat := 37886
def frameStart : Nat := 37847
def rule : BoundRule := .identity (.predecessor 0 37885 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37885 .coefficient)
      LeftAuthority37883.bound (LeftAuthority37883.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events147.exact37884RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37883.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37883.derived selector witness)

def rawBound : CoeffClass := LeftAuthority37883.bound
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority37883.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority37883.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound37886

namespace LeftBound37903
def owner : Owner := ⟨.program ⟨214⟩, ⟨16716⟩⟩
def transferEvent : Nat := 37903
def frameStart : Nat := 37847
def rule : BoundRule := .sum [.predecessor 0 37901 .coefficient, .predecessor 1 37902 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37901 .coefficient)
      LeftBound37886.bound (LeftBound37886.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound37886.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37902 .coefficient)
      LeftAuthority37899.bound (LeftAuthority37899.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority37899.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound37886.bound, LeftAuthority37899.bound]
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37886.bound, LeftAuthority37899.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound37886.actual selector witness, LeftAuthority37899.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound37903

namespace LeftBound37906
def owner : Owner := ⟨.program ⟨214⟩, ⟨16717⟩⟩
def transferEvent : Nat := 37906
def frameStart : Nat := 37847
def rule : BoundRule := .identity (.predecessor 0 37905 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37905 .coefficient)
      LeftBound37903.bound (LeftBound37903.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound37903.derived selector witness)

def rawBound : CoeffClass := LeftBound37903.bound
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37903.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound37903.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound37906

namespace LeftBound37912
def owner : Owner := ⟨.program ⟨214⟩, ⟨16718⟩⟩
def transferEvent : Nat := 37912
def frameStart : Nat := 37847
def rule : BoundRule := .product (.predecessor 0 37910 .coefficient) (.predecessor 1 37911 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37910 .coefficient)
      LeftAuthority37908.bound (LeftAuthority37908.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events148.exact37909RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37908.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37908.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37911 .coefficient)
      LeftBound37906.bound (LeftBound37906.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events148.exact37907RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37906.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37906.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority37908.bound LeftBound37906.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority37908.bound, LeftBound37906.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority37908.actual selector witness) * (LeftBound37906.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound37912

namespace LeftBound37920
def owner : Owner := ⟨.program ⟨214⟩, ⟨16719⟩⟩
def transferEvent : Nat := 37920
def frameStart : Nat := 37847
def rule : BoundRule := .sum [.predecessor 0 37918 .coefficient, .predecessor 1 37919 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37918 .coefficient)
      LeftAuthority37916.bound (LeftAuthority37916.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events148.exact37917RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37916.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37916.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37919 .coefficient)
      LeftBound37912.bound (LeftBound37912.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events148.exact37914RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37912.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37912.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority37916.bound, LeftBound37912.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority37916.bound, LeftBound37912.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority37916.actual selector witness, LeftBound37912.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound37920

namespace LeftBound37924
def owner : Owner := ⟨.program ⟨214⟩, ⟨29412⟩⟩
def transferEvent : Nat := 37924
def frameStart : Nat := 37847
def rule : BoundRule := .product (.predecessor 0 37922 .coefficient) (.predecessor 1 37923 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37922 .coefficient)
      LeftBound37920.bound (LeftBound37920.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events148.exact37921RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37920.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37920.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37923 .coefficient)
      LeftAuthority37897.bound (LeftAuthority37897.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events148.exact37898RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37897.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37897.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound37920.bound LeftAuthority37897.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37920.bound, LeftAuthority37897.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound37920.actual selector witness) * (LeftAuthority37897.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound37924

namespace LeftBound37935
def owner : Owner := ⟨.program ⟨214⟩, ⟨16686⟩⟩
def transferEvent : Nat := 37935
def frameStart : Nat := 37847
def rule : BoundRule := .product (.predecessor 0 37933 .coefficient) (.predecessor 1 37934 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37933 .coefficient)
      LeftAuthority37908.bound (LeftAuthority37908.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events148.exact37909RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37908.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37908.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37934 .coefficient)
      LeftAuthority37931.bound (LeftAuthority37931.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events148.exact37932RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37931.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37931.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority37908.bound LeftAuthority37931.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority37908.bound, LeftAuthority37931.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority37908.actual selector witness) * (LeftAuthority37931.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound37935

namespace LeftBound37943
def owner : Owner := ⟨.program ⟨214⟩, ⟨16687⟩⟩
def transferEvent : Nat := 37943
def frameStart : Nat := 37847
def rule : BoundRule := .sum [.predecessor 0 37941 .coefficient, .predecessor 1 37942 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37941 .coefficient)
      LeftAuthority37939.bound (LeftAuthority37939.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events148.exact37940RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37939.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37939.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37942 .coefficient)
      LeftBound37935.bound (LeftBound37935.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events148.exact37937RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37935.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37935.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority37939.bound, LeftBound37935.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority37939.bound, LeftBound37935.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority37939.actual selector witness, LeftBound37935.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound37943

namespace LeftBound37947
def owner : Owner := ⟨.program ⟨214⟩, ⟨29416⟩⟩
def transferEvent : Nat := 37947
def frameStart : Nat := 37847
def rule : BoundRule := .sum [.predecessor 0 37945 .coefficient, .predecessor 1 37946 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37945 .coefficient)
      LeftBound37943.bound (LeftBound37943.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events148.exact37944RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37943.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37943.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37946 .coefficient)
      LeftBound37924.bound (LeftBound37924.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events148.exact37929RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37924.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37924.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound37943.bound, LeftBound37924.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37943.bound, LeftBound37924.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound37943.actual selector witness, LeftBound37924.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound37947

namespace LeftBound37960
def owner : Owner := ⟨.program ⟨214⟩, ⟨29414⟩⟩
def transferEvent : Nat := 37960
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 37958 .coefficient, .predecessor 1 37959 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37958 .coefficient)
      LeftBound37789.bound (LeftBound37789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events148.exact37957RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37789.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37789.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37959 .coefficient)
      LeftBound37772.bound (LeftBound37772.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events147.exact37779RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37772.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37772.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound37789.bound, LeftBound37772.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37789.bound, LeftBound37772.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound37789.actual selector witness, LeftBound37772.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound37960

namespace LeftBound37963
def owner : Owner := ⟨.program ⟨214⟩, ⟨29414⟩⟩
def transferEvent : Nat := 37963
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 37957 .summary, .result 37779 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 37957 .summary)
      LeftBound37791.bound (LeftBound37791.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22419⟩⟩) (rawTerms := some (Proof.Events148.exact37957RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound37791.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 37779 .summary)
      LeftBound37774.bound (LeftBound37774.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29413⟩⟩) (rawTerms := some (Proof.Events147.exact37779RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound37774.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound37791.bound, LeftBound37774.bound]
def bound : CoeffClass := .finite ⟨1292382248169874534400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37791.bound, LeftBound37774.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound37791.actual selector witness, LeftBound37774.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound37963

namespace LeftBound37987
def owner : Owner := ⟨.program ⟨214⟩, ⟨12585⟩⟩
def transferEvent : Nat := 37987
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 37985 .coefficient) (.predecessor 1 37986 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37985 .coefficient)
      LeftAuthority1681.bound (LeftAuthority1681.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events006.exact1682RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1681.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1681.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37986 .coefficient)
      LeftBound36043.bound (LeftBound36043.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact36045RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36043.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36043.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority1681.bound LeftBound36043.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1681.bound, LeftBound36043.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority1681.actual selector witness) * (LeftBound36043.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound37987

namespace LeftBound37992
def owner : Owner := ⟨.program ⟨214⟩, ⟨7318⟩⟩
def transferEvent : Nat := 37992
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 37990 .coefficient) (.predecessor 1 37991 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37990 .coefficient)
      LeftBound35914.bound (LeftBound35914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact35915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35914.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35914.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37991 .coefficient)
      LeftBound8475.bound (LeftBound8475.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events033.exact8476RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8475.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8475.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound35914.bound LeftBound8475.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35914.bound, LeftBound8475.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound35914.actual selector witness) * (LeftBound8475.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound37992

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
