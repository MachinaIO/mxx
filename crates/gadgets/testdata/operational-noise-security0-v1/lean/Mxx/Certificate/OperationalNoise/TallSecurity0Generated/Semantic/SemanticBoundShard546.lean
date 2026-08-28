import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard033
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard545

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound80456
def owner : Owner := ⟨.program ⟨214⟩, ⟨10243⟩⟩
def transferEvent : Nat := 80456
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 80454 .coefficient, .predecessor 1 80455 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80454 .coefficient)
      LeftBound80452.bound (LeftBound80452.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events314.exact80453RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80452.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80452.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80455 .coefficient)
      LeftBound7005.bound (LeftBound7005.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact7006RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7005.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7005.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound80452.bound, LeftBound7005.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80452.bound, LeftBound7005.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound80452.actual selector witness, LeftBound7005.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound80456

namespace LeftBound80457
def owner : Owner := ⟨.program ⟨214⟩, ⟨10243⟩⟩
def transferEvent : Nat := 80457
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨83⟩⟩]⟩ [⟨.result 7006 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7006 .coefficient)
      LeftBound7005.bound (LeftBound7005.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨83⟩⟩) (rawTerms := some (Proof.Events027.exact7006RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7005.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7005.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound7005.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7005.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound7005.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound80457

namespace LeftBound80462
def owner : Owner := ⟨.program ⟨214⟩, ⟨10244⟩⟩
def transferEvent : Nat := 80462
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 80460 .coefficient) (.predecessor 1 80461 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80460 .coefficient)
      LeftBound80456.bound (LeftBound80456.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events314.exact80459RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80456.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80456.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80461 .coefficient)
      LeftBound7002.bound (LeftBound7002.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact7003RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7002.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7002.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound80456.bound LeftBound7002.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80456.bound, LeftBound7002.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound80456.actual selector witness) * (LeftBound7002.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound80462

namespace LeftBound80463
def owner : Owner := ⟨.program ⟨214⟩, ⟨10244⟩⟩
def transferEvent : Nat := 80463
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩ [⟨.result 6999 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6999 .coefficient)
      LeftAuthority6998.bound (LeftAuthority6998.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7879⟩⟩) (rawTerms := some (Proof.Events027.exact6999RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6998.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6998.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority6998.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6998.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority6998.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound80463

namespace LeftBound80464
def owner : Owner := ⟨.program ⟨214⟩, ⟨10244⟩⟩
def transferEvent : Nat := 80464
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80459 .summary) (.transfer 80463) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80459 .summary)
      LeftBound80457.bound (LeftBound80457.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10243⟩⟩) (rawTerms := some (Proof.Events314.exact80459RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80457.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 80463)
      LeftBound80463.bound (LeftBound80463.actual selector witness) := by
  exact .transfer (LeftBound80463.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound80457.bound LeftBound80463.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80457.bound, LeftBound80463.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound80457.actual selector witness) * (LeftBound80463.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound80464

namespace LeftBound80472
def owner : Owner := ⟨.program ⟨214⟩, ⟨13161⟩⟩
def transferEvent : Nat := 80472
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 80470 .coefficient, .predecessor 1 80471 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80470 .coefficient)
      LeftBound80462.bound (LeftBound80462.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events314.exact80469RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80462.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80462.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80471 .coefficient)
      LeftBound80434.bound (LeftBound80434.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events314.exact80439RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80434.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80434.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound80462.bound, LeftBound80434.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80462.bound, LeftBound80434.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound80462.actual selector witness, LeftBound80434.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound80472

namespace LeftBound80474
def owner : Owner := ⟨.program ⟨214⟩, ⟨13161⟩⟩
def transferEvent : Nat := 80474
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 80469 .summary, .result 80439 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80469 .summary)
      LeftBound80464.bound (LeftBound80464.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10244⟩⟩) (rawTerms := some (Proof.Events314.exact80469RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80464.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80439 .summary)
      LeftBound80436.bound (LeftBound80436.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13160⟩⟩) (rawTerms := some (Proof.Events314.exact80439RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80436.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound80464.bound, LeftBound80436.bound]
def bound : CoeffClass := .finite ⟨95468672, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80464.bound, LeftBound80436.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound80464.actual selector witness, LeftBound80436.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound80474

namespace LeftBound80478
def owner : Owner := ⟨.program ⟨214⟩, ⟨25682⟩⟩
def transferEvent : Nat := 80478
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 80476 .coefficient) (.predecessor 1 80477 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80476 .coefficient)
      LeftBound80472.bound (LeftBound80472.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events314.exact80475RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80472.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80472.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80477 .coefficient)
      LeftAuthority80410.bound (LeftAuthority80410.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events314.exact80411RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80410.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80410.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound80472.bound LeftAuthority80410.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80472.bound, LeftAuthority80410.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound80472.actual selector witness) * (LeftAuthority80410.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound80478

namespace LeftBound80479
def owner : Owner := ⟨.program ⟨214⟩, ⟨25682⟩⟩
def transferEvent : Nat := 80479
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25681⟩⟩]⟩ [⟨.result 80411 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80411 .coefficient)
      LeftAuthority80410.bound (LeftAuthority80410.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25681⟩⟩) (rawTerms := some (Proof.Events314.exact80411RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80410.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80410.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority80410.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority80410.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority80410.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound80479

namespace LeftBound80480
def owner : Owner := ⟨.program ⟨214⟩, ⟨25682⟩⟩
def transferEvent : Nat := 80480
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80475 .summary) (.transfer 80479) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80475 .summary)
      LeftBound80474.bound (LeftBound80474.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13161⟩⟩) (rawTerms := some (Proof.Events314.exact80475RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80474.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 80479)
      LeftBound80479.bound (LeftBound80479.actual selector witness) := by
  exact .transfer (LeftBound80479.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound80474.bound LeftBound80479.bound
def bound : CoeffClass := .finite ⟨350371553738752, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80474.bound, LeftBound80479.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound80474.actual selector witness) * (LeftBound80479.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound80480

namespace LeftBound80491
def owner : Owner := ⟨.program ⟨214⟩, ⟨20178⟩⟩
def transferEvent : Nat := 80491
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 80489 .coefficient) (.value (.predecessor 1 80490 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80489 .coefficient)
      LeftAuthority80487.bound (LeftAuthority80487.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events314.exact80488RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80487.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80487.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80490 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority80487.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority80487.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority80487.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound80491

namespace LeftBound80495
def owner : Owner := ⟨.program ⟨214⟩, ⟨20179⟩⟩
def transferEvent : Nat := 80495
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 80493 .coefficient) (.predecessor 1 80494 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80493 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80494 .coefficient)
      LeftBound80491.bound (LeftBound80491.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events314.exact80492RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80491.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80491.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound80491.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound80491.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound80491.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound80495

namespace LeftBound80496
def owner : Owner := ⟨.program ⟨214⟩, ⟨20179⟩⟩
def transferEvent : Nat := 80496
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20176⟩⟩]⟩ [⟨.result 80488 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80488 .coefficient)
      LeftAuthority80487.bound (LeftAuthority80487.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20176⟩⟩) (rawTerms := some (Proof.Events314.exact80488RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80487.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80487.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority80487.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority80487.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority80487.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound80496

namespace LeftBound80497
def owner : Owner := ⟨.program ⟨214⟩, ⟨20179⟩⟩
def transferEvent : Nat := 80497
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 80496) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 80496)
      LeftBound80496.bound (LeftBound80496.actual selector witness) := by
  exact .transfer (LeftBound80496.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound80496.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound80496.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound80496.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound80497

namespace LeftBound80576
def owner : Owner := ⟨.program ⟨214⟩, ⟨13155⟩⟩
def transferEvent : Nat := 80576
def frameStart : Nat := 80547
def rule : BoundRule := .product (.predecessor 0 80574 .coefficient) (.predecessor 1 80575 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80574 .coefficient)
      LeftAuthority80572.bound (LeftAuthority80572.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events314.exact80573RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80572.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80572.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80575 .coefficient)
      LeftAuthority80569.bound (LeftAuthority80569.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events314.exact80570RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80569.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80569.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority80572.bound LeftAuthority80569.bound
def bound : CoeffClass := .finite ⟨3364, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority80572.bound, LeftAuthority80569.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority80572.actual selector witness) * (LeftAuthority80569.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound80576

namespace LeftBound80580
def owner : Owner := ⟨.program ⟨214⟩, ⟨13156⟩⟩
def transferEvent : Nat := 80580
def frameStart : Nat := 80547
def rule : BoundRule := .identity (.predecessor 0 80579 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80579 .coefficient)
      LeftBound80576.bound (LeftBound80576.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events314.exact80578RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80576.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80576.derived selector witness)

def rawBound : CoeffClass := LeftBound80576.bound
def bound : CoeffClass := .finite ⟨3364, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80576.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound80576.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound80580

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
