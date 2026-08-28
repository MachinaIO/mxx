import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard018
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard206

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound31744
def owner : Owner := ⟨.program ⟨214⟩, ⟨22783⟩⟩
def transferEvent : Nat := 31744
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 31742 .coefficient) (.predecessor 1 31743 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31742 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31743 .coefficient)
      LeftBound31740.bound (LeftBound31740.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31741RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31740.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31740.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound31740.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound31740.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound31740.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound31744

namespace LeftBound31745
def owner : Owner := ⟨.program ⟨214⟩, ⟨22783⟩⟩
def transferEvent : Nat := 31745
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22780⟩⟩]⟩ [⟨.result 31737 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 31737 .coefficient)
      LeftAuthority31736.bound (LeftAuthority31736.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22780⟩⟩) (rawTerms := some (Proof.Events123.exact31737RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31736.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31736.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority31736.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority31736.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority31736.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound31745

namespace LeftBound31746
def owner : Owner := ⟨.program ⟨214⟩, ⟨22783⟩⟩
def transferEvent : Nat := 31746
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 31745) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 31745)
      LeftBound31745.bound (LeftBound31745.actual selector witness) := by
  exact .transfer (LeftBound31745.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound31745.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound31745.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound31745.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound31746

namespace LeftBound31841
def owner : Owner := ⟨.program ⟨214⟩, ⟨17024⟩⟩
def transferEvent : Nat := 31841
def frameStart : Nat := 31802
def rule : BoundRule := .identity (.predecessor 0 31840 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31840 .coefficient)
      LeftAuthority31838.bound (LeftAuthority31838.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events124.exact31839RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31838.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31838.derived selector witness)

def rawBound : CoeffClass := LeftAuthority31838.bound
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority31838.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority31838.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound31841

namespace LeftBound31858
def owner : Owner := ⟨.program ⟨214⟩, ⟨17063⟩⟩
def transferEvent : Nat := 31858
def frameStart : Nat := 31802
def rule : BoundRule := .sum [.predecessor 0 31856 .coefficient, .predecessor 1 31857 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31856 .coefficient)
      LeftBound31841.bound (LeftBound31841.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound31841.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31857 .coefficient)
      LeftAuthority31854.bound (LeftAuthority31854.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority31854.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31841.bound, LeftAuthority31854.bound]
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31841.bound, LeftAuthority31854.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound31841.actual selector witness, LeftAuthority31854.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31858

namespace LeftBound31861
def owner : Owner := ⟨.program ⟨214⟩, ⟨17064⟩⟩
def transferEvent : Nat := 31861
def frameStart : Nat := 31802
def rule : BoundRule := .identity (.predecessor 0 31860 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31860 .coefficient)
      LeftBound31858.bound (LeftBound31858.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound31858.derived selector witness)

def rawBound : CoeffClass := LeftBound31858.bound
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31858.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound31858.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound31861

namespace LeftBound31867
def owner : Owner := ⟨.program ⟨214⟩, ⟨17065⟩⟩
def transferEvent : Nat := 31867
def frameStart : Nat := 31802
def rule : BoundRule := .product (.predecessor 0 31865 .coefficient) (.predecessor 1 31866 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31865 .coefficient)
      LeftAuthority31863.bound (LeftAuthority31863.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events124.exact31864RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31863.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31863.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31866 .coefficient)
      LeftBound31861.bound (LeftBound31861.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events124.exact31862RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31861.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31861.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority31863.bound LeftBound31861.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority31863.bound, LeftBound31861.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority31863.actual selector witness) * (LeftBound31861.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound31867

namespace LeftBound31875
def owner : Owner := ⟨.program ⟨214⟩, ⟨17066⟩⟩
def transferEvent : Nat := 31875
def frameStart : Nat := 31802
def rule : BoundRule := .sum [.predecessor 0 31873 .coefficient, .predecessor 1 31874 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31873 .coefficient)
      LeftAuthority31871.bound (LeftAuthority31871.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events124.exact31872RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31871.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31871.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31874 .coefficient)
      LeftBound31867.bound (LeftBound31867.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events124.exact31869RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31867.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31867.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority31871.bound, LeftBound31867.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority31871.bound, LeftBound31867.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority31871.actual selector witness, LeftBound31867.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31875

namespace LeftBound31879
def owner : Owner := ⟨.program ⟨214⟩, ⟨30177⟩⟩
def transferEvent : Nat := 31879
def frameStart : Nat := 31802
def rule : BoundRule := .product (.predecessor 0 31877 .coefficient) (.predecessor 1 31878 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31877 .coefficient)
      LeftBound31875.bound (LeftBound31875.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events124.exact31876RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31875.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31875.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31878 .coefficient)
      LeftAuthority31852.bound (LeftAuthority31852.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events124.exact31853RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31852.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31852.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound31875.bound LeftAuthority31852.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31875.bound, LeftAuthority31852.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound31875.actual selector witness) * (LeftAuthority31852.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound31879

namespace LeftBound31890
def owner : Owner := ⟨.program ⟨214⟩, ⟨18138⟩⟩
def transferEvent : Nat := 31890
def frameStart : Nat := 31802
def rule : BoundRule := .product (.predecessor 0 31888 .coefficient) (.predecessor 1 31889 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31888 .coefficient)
      LeftAuthority31863.bound (LeftAuthority31863.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events124.exact31864RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31863.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31863.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31889 .coefficient)
      LeftAuthority31886.bound (LeftAuthority31886.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events124.exact31887RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31886.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31886.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority31863.bound LeftAuthority31886.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority31863.bound, LeftAuthority31886.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority31863.actual selector witness) * (LeftAuthority31886.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound31890

namespace LeftBound31898
def owner : Owner := ⟨.program ⟨214⟩, ⟨18139⟩⟩
def transferEvent : Nat := 31898
def frameStart : Nat := 31802
def rule : BoundRule := .sum [.predecessor 0 31896 .coefficient, .predecessor 1 31897 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31896 .coefficient)
      LeftAuthority31894.bound (LeftAuthority31894.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events124.exact31895RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31894.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31894.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31897 .coefficient)
      LeftBound31890.bound (LeftBound31890.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events124.exact31892RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31890.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31890.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority31894.bound, LeftBound31890.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority31894.bound, LeftBound31890.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority31894.actual selector witness, LeftBound31890.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31898

namespace LeftBound31902
def owner : Owner := ⟨.program ⟨214⟩, ⟨30182⟩⟩
def transferEvent : Nat := 31902
def frameStart : Nat := 31802
def rule : BoundRule := .sum [.predecessor 0 31900 .coefficient, .predecessor 1 31901 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31900 .coefficient)
      LeftBound31898.bound (LeftBound31898.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events124.exact31899RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31898.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31898.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31901 .coefficient)
      LeftBound31879.bound (LeftBound31879.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events124.exact31884RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31879.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31879.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31898.bound, LeftBound31879.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31898.bound, LeftBound31879.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound31898.actual selector witness, LeftBound31879.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31902

namespace LeftBound31915
def owner : Owner := ⟨.program ⟨214⟩, ⟨30179⟩⟩
def transferEvent : Nat := 31915
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 31913 .coefficient, .predecessor 1 31914 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31913 .coefficient)
      LeftBound31744.bound (LeftBound31744.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events124.exact31912RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31744.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31744.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31914 .coefficient)
      LeftBound31727.bound (LeftBound31727.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31734RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31727.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31727.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31744.bound, LeftBound31727.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31744.bound, LeftBound31727.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound31744.actual selector witness, LeftBound31727.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31915

namespace LeftBound31918
def owner : Owner := ⟨.program ⟨214⟩, ⟨30179⟩⟩
def transferEvent : Nat := 31918
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 31912 .summary, .result 31734 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 31912 .summary)
      LeftBound31746.bound (LeftBound31746.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22783⟩⟩) (rawTerms := some (Proof.Events124.exact31912RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound31746.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 31734 .summary)
      LeftBound31729.bound (LeftBound31729.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30178⟩⟩) (rawTerms := some (Proof.Events123.exact31734RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound31729.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31746.bound, LeftBound31729.bound]
def bound : CoeffClass := .finite ⟨1292539135285018636288, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31746.bound, LeftBound31729.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound31746.actual selector witness, LeftBound31729.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31918

namespace LeftBound31922
def owner : Owner := ⟨.program ⟨214⟩, ⟨30180⟩⟩
def transferEvent : Nat := 31922
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 31920 .coefficient) (.predecessor 1 31921 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31920 .coefficient)
      LeftBound31915.bound (LeftBound31915.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events124.exact31919RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31915.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31915.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31921 .coefficient)
      LeftBound5518.bound (LeftBound5518.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5519RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5518.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5518.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound31915.bound LeftBound5518.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31915.bound, LeftBound5518.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound31915.actual selector witness) * (LeftBound5518.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound31922

namespace LeftBound31923
def owner : Owner := ⟨.program ⟨214⟩, ⟨30180⟩⟩
def transferEvent : Nat := 31923
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6657⟩⟩]⟩ [⟨.result 5515 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5515 .coefficient)
      LeftAuthority5514.bound (LeftAuthority5514.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6657⟩⟩) (rawTerms := some (Proof.Events021.exact5515RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5514.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5514.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5514.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5514.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5514.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound31923

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
