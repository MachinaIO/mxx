import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard442
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard510

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound75587
def owner : Owner := ⟨.program ⟨214⟩, ⟨30102⟩⟩
def transferEvent : Nat := 75587
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 75582 .summary) (.transfer 75586) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 75582 .summary)
      LeftBound75581.bound (LeftBound75581.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30101⟩⟩) (rawTerms := some (Proof.Events295.exact75582RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound75581.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 75586)
      LeftBound75586.bound (LeftBound75586.actual selector witness) := by
  exact .transfer (LeftBound75586.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound75581.bound LeftBound75586.bound
def bound : CoeffClass := .finite ⟨313276371396785701094268180805713920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75581.bound, LeftBound75586.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound75581.actual selector witness) * (LeftBound75586.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound75587

namespace LeftBound75602
def owner : Owner := ⟨.program ⟨214⟩, ⟨30090⟩⟩
def transferEvent : Nat := 75602
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 75600 .coefficient) (.predecessor 1 75601 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75600 .coefficient)
      LeftBound65569.bound (LeftBound65569.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65573RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65569.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65569.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75601 .coefficient)
      LeftAuthority75598.bound (LeftAuthority75598.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events295.exact75599RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75598.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75598.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound65569.bound LeftAuthority75598.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65569.bound, LeftAuthority75598.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound65569.actual selector witness) * (LeftAuthority75598.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound75602

namespace LeftBound75603
def owner : Owner := ⟨.program ⟨214⟩, ⟨30090⟩⟩
def transferEvent : Nat := 75603
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨30088⟩⟩]⟩ [⟨.result 75599 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 75599 .coefficient)
      LeftAuthority75598.bound (LeftAuthority75598.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨30088⟩⟩) (rawTerms := some (Proof.Events295.exact75599RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75598.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75598.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority75598.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority75598.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority75598.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound75603

namespace LeftBound75604
def owner : Owner := ⟨.program ⟨214⟩, ⟨30090⟩⟩
def transferEvent : Nat := 75604
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65573 .summary) (.transfer 75603) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65573 .summary)
      LeftBound65572.bound (LeftBound65572.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25755⟩⟩) (rawTerms := some (Proof.Events256.exact65573RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65572.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 75603)
      LeftBound75603.bound (LeftBound75603.actual selector witness) := by
  exact .transfer (LeftBound75603.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound65572.bound LeftBound75603.bound
def bound : CoeffClass := .finite ⟨1292539133473715126272, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65572.bound, LeftBound75603.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound65572.actual selector witness) * (LeftBound75603.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound75604

namespace LeftBound75615
def owner : Owner := ⟨.program ⟨214⟩, ⟨22766⟩⟩
def transferEvent : Nat := 75615
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 75613 .coefficient) (.value (.predecessor 1 75614 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75613 .coefficient)
      LeftAuthority75611.bound (LeftAuthority75611.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events295.exact75612RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75611.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75611.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75614 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority75611.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority75611.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority75611.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound75615

namespace LeftBound75619
def owner : Owner := ⟨.program ⟨214⟩, ⟨22767⟩⟩
def transferEvent : Nat := 75619
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 75617 .coefficient) (.predecessor 1 75618 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75617 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75618 .coefficient)
      LeftBound75615.bound (LeftBound75615.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events295.exact75616RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75615.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75615.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound75615.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound75615.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound75615.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound75619

namespace LeftBound75620
def owner : Owner := ⟨.program ⟨214⟩, ⟨22767⟩⟩
def transferEvent : Nat := 75620
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22764⟩⟩]⟩ [⟨.result 75612 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 75612 .coefficient)
      LeftAuthority75611.bound (LeftAuthority75611.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22764⟩⟩) (rawTerms := some (Proof.Events295.exact75612RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75611.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75611.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority75611.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority75611.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority75611.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound75620

namespace LeftBound75621
def owner : Owner := ⟨.program ⟨214⟩, ⟨22767⟩⟩
def transferEvent : Nat := 75621
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 75620) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 75620)
      LeftBound75620.bound (LeftBound75620.actual selector witness) := by
  exact .transfer (LeftBound75620.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound75620.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound75620.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound75620.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound75621

namespace LeftBound75716
def owner : Owner := ⟨.program ⟨214⟩, ⟨17008⟩⟩
def transferEvent : Nat := 75716
def frameStart : Nat := 75677
def rule : BoundRule := .identity (.predecessor 0 75715 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75715 .coefficient)
      LeftAuthority75713.bound (LeftAuthority75713.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events295.exact75714RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75713.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75713.derived selector witness)

def rawBound : CoeffClass := LeftAuthority75713.bound
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority75713.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority75713.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound75716

namespace LeftBound75733
def owner : Owner := ⟨.program ⟨214⟩, ⟨17047⟩⟩
def transferEvent : Nat := 75733
def frameStart : Nat := 75677
def rule : BoundRule := .sum [.predecessor 0 75731 .coefficient, .predecessor 1 75732 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75731 .coefficient)
      LeftBound75716.bound (LeftBound75716.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound75716.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75732 .coefficient)
      LeftAuthority75729.bound (LeftAuthority75729.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority75729.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75716.bound, LeftAuthority75729.bound]
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75716.bound, LeftAuthority75729.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound75716.actual selector witness, LeftAuthority75729.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75733

namespace LeftBound75736
def owner : Owner := ⟨.program ⟨214⟩, ⟨17048⟩⟩
def transferEvent : Nat := 75736
def frameStart : Nat := 75677
def rule : BoundRule := .identity (.predecessor 0 75735 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75735 .coefficient)
      LeftBound75733.bound (LeftBound75733.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound75733.derived selector witness)

def rawBound : CoeffClass := LeftBound75733.bound
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75733.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound75733.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound75736

namespace LeftBound75742
def owner : Owner := ⟨.program ⟨214⟩, ⟨17049⟩⟩
def transferEvent : Nat := 75742
def frameStart : Nat := 75677
def rule : BoundRule := .product (.predecessor 0 75740 .coefficient) (.predecessor 1 75741 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75740 .coefficient)
      LeftAuthority75738.bound (LeftAuthority75738.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events295.exact75739RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75738.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75738.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75741 .coefficient)
      LeftBound75736.bound (LeftBound75736.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events295.exact75737RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75736.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75736.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority75738.bound LeftBound75736.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority75738.bound, LeftBound75736.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority75738.actual selector witness) * (LeftBound75736.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound75742

namespace LeftBound75750
def owner : Owner := ⟨.program ⟨214⟩, ⟨17050⟩⟩
def transferEvent : Nat := 75750
def frameStart : Nat := 75677
def rule : BoundRule := .sum [.predecessor 0 75748 .coefficient, .predecessor 1 75749 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75748 .coefficient)
      LeftAuthority75746.bound (LeftAuthority75746.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events295.exact75747RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75746.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75746.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75749 .coefficient)
      LeftBound75742.bound (LeftBound75742.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events295.exact75744RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75742.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75742.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority75746.bound, LeftBound75742.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority75746.bound, LeftBound75742.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority75746.actual selector witness, LeftBound75742.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75750

namespace LeftBound75754
def owner : Owner := ⟨.program ⟨214⟩, ⟨30089⟩⟩
def transferEvent : Nat := 75754
def frameStart : Nat := 75677
def rule : BoundRule := .product (.predecessor 0 75752 .coefficient) (.predecessor 1 75753 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75752 .coefficient)
      LeftBound75750.bound (LeftBound75750.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events295.exact75751RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75750.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75750.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75753 .coefficient)
      LeftAuthority75727.bound (LeftAuthority75727.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events295.exact75728RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75727.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75727.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound75750.bound LeftAuthority75727.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75750.bound, LeftAuthority75727.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound75750.actual selector witness) * (LeftAuthority75727.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound75754

namespace LeftBound75765
def owner : Owner := ⟨.program ⟨214⟩, ⟨18122⟩⟩
def transferEvent : Nat := 75765
def frameStart : Nat := 75677
def rule : BoundRule := .product (.predecessor 0 75763 .coefficient) (.predecessor 1 75764 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75763 .coefficient)
      LeftAuthority75738.bound (LeftAuthority75738.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events295.exact75739RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75738.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75738.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75764 .coefficient)
      LeftAuthority75761.bound (LeftAuthority75761.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events295.exact75762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75761.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75761.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority75738.bound LeftAuthority75761.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority75738.bound, LeftAuthority75761.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority75738.actual selector witness) * (LeftAuthority75761.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound75765

namespace LeftBound75773
def owner : Owner := ⟨.program ⟨214⟩, ⟨18123⟩⟩
def transferEvent : Nat := 75773
def frameStart : Nat := 75677
def rule : BoundRule := .sum [.predecessor 0 75771 .coefficient, .predecessor 1 75772 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75771 .coefficient)
      LeftAuthority75769.bound (LeftAuthority75769.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events295.exact75770RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75769.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75769.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75772 .coefficient)
      LeftBound75765.bound (LeftBound75765.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events295.exact75767RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75765.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75765.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority75769.bound, LeftBound75765.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority75769.bound, LeftBound75765.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority75769.actual selector witness, LeftBound75765.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75773

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
