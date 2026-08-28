import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard286

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound42592
def owner : Owner := ⟨.program ⟨214⟩, ⟨27243⟩⟩
def transferEvent : Nat := 42592
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 42590 .coefficient) (.predecessor 1 42591 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42590 .coefficient)
      LeftBound42585.bound (LeftBound42585.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events166.exact42589RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42585.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42585.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42591 .coefficient)
      LeftAuthority42311.bound (LeftAuthority42311.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events165.exact42312RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42311.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42311.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound42585.bound LeftAuthority42311.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42585.bound, LeftAuthority42311.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound42585.actual selector witness) * (LeftAuthority42311.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound42592

namespace LeftBound42593
def owner : Owner := ⟨.program ⟨214⟩, ⟨27243⟩⟩
def transferEvent : Nat := 42593
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27241⟩⟩]⟩ [⟨.result 42312 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 42312 .coefficient)
      LeftAuthority42311.bound (LeftAuthority42311.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27241⟩⟩) (rawTerms := some (Proof.Events165.exact42312RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42311.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42311.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority42311.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority42311.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority42311.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound42593

namespace LeftBound42594
def owner : Owner := ⟨.program ⟨214⟩, ⟨27243⟩⟩
def transferEvent : Nat := 42594
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 42589 .summary) (.transfer 42593) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 42589 .summary)
      LeftBound42588.bound (LeftBound42588.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25847⟩⟩) (rawTerms := some (Proof.Events166.exact42589RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound42588.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 42593)
      LeftBound42593.bound (LeftBound42593.actual selector witness) := by
  exact .transfer (LeftBound42593.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound42588.bound LeftBound42593.bound
def bound : CoeffClass := .finite ⟨1291978822348200476672, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42588.bound, LeftBound42593.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound42588.actual selector witness) * (LeftBound42593.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound42594

namespace LeftBound42605
def owner : Owner := ⟨.program ⟨214⟩, ⟨20978⟩⟩
def transferEvent : Nat := 42605
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 42603 .coefficient) (.value (.predecessor 1 42604 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42603 .coefficient)
      LeftAuthority42601.bound (LeftAuthority42601.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events166.exact42602RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42601.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42601.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42604 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority42601.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority42601.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority42601.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound42605

namespace LeftBound42609
def owner : Owner := ⟨.program ⟨214⟩, ⟨20979⟩⟩
def transferEvent : Nat := 42609
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 42607 .coefficient) (.predecessor 1 42608 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42607 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42608 .coefficient)
      LeftBound42605.bound (LeftBound42605.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events166.exact42606RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42605.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42605.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound42605.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound42605.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound42605.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound42609

namespace LeftBound42610
def owner : Owner := ⟨.program ⟨214⟩, ⟨20979⟩⟩
def transferEvent : Nat := 42610
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20976⟩⟩]⟩ [⟨.result 42602 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 42602 .coefficient)
      LeftAuthority42601.bound (LeftAuthority42601.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20976⟩⟩) (rawTerms := some (Proof.Events166.exact42602RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42601.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42601.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority42601.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority42601.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority42601.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound42610

namespace LeftBound42611
def owner : Owner := ⟨.program ⟨214⟩, ⟨20979⟩⟩
def transferEvent : Nat := 42611
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 42610) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 42610)
      LeftBound42610.bound (LeftBound42610.actual selector witness) := by
  exact .transfer (LeftBound42610.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound42610.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound42610.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound42610.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound42611

namespace LeftBound42706
def owner : Owner := ⟨.program ⟨214⟩, ⟨15592⟩⟩
def transferEvent : Nat := 42706
def frameStart : Nat := 42667
def rule : BoundRule := .identity (.predecessor 0 42705 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42705 .coefficient)
      LeftAuthority42703.bound (LeftAuthority42703.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events166.exact42704RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42703.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42703.derived selector witness)

def rawBound : CoeffClass := LeftAuthority42703.bound
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority42703.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority42703.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound42706

namespace LeftBound42723
def owner : Owner := ⟨.program ⟨214⟩, ⟨15666⟩⟩
def transferEvent : Nat := 42723
def frameStart : Nat := 42667
def rule : BoundRule := .sum [.predecessor 0 42721 .coefficient, .predecessor 1 42722 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42721 .coefficient)
      LeftBound42706.bound (LeftBound42706.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound42706.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42722 .coefficient)
      LeftAuthority42719.bound (LeftAuthority42719.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority42719.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound42706.bound, LeftAuthority42719.bound]
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42706.bound, LeftAuthority42719.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound42706.actual selector witness, LeftAuthority42719.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound42723

namespace LeftBound42726
def owner : Owner := ⟨.program ⟨214⟩, ⟨15667⟩⟩
def transferEvent : Nat := 42726
def frameStart : Nat := 42667
def rule : BoundRule := .identity (.predecessor 0 42725 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42725 .coefficient)
      LeftBound42723.bound (LeftBound42723.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound42723.derived selector witness)

def rawBound : CoeffClass := LeftBound42723.bound
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42723.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound42723.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound42726

namespace LeftBound42732
def owner : Owner := ⟨.program ⟨214⟩, ⟨15668⟩⟩
def transferEvent : Nat := 42732
def frameStart : Nat := 42667
def rule : BoundRule := .product (.predecessor 0 42730 .coefficient) (.predecessor 1 42731 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42730 .coefficient)
      LeftAuthority42728.bound (LeftAuthority42728.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events166.exact42729RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42728.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42728.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42731 .coefficient)
      LeftBound42726.bound (LeftBound42726.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events166.exact42727RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42726.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42726.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority42728.bound LeftBound42726.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority42728.bound, LeftBound42726.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority42728.actual selector witness) * (LeftBound42726.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound42732

namespace LeftBound42740
def owner : Owner := ⟨.program ⟨214⟩, ⟨15669⟩⟩
def transferEvent : Nat := 42740
def frameStart : Nat := 42667
def rule : BoundRule := .sum [.predecessor 0 42738 .coefficient, .predecessor 1 42739 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42738 .coefficient)
      LeftAuthority42736.bound (LeftAuthority42736.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events166.exact42737RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42736.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42736.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42739 .coefficient)
      LeftBound42732.bound (LeftBound42732.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events166.exact42734RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42732.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42732.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority42736.bound, LeftBound42732.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority42736.bound, LeftBound42732.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority42736.actual selector witness, LeftBound42732.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound42740

namespace LeftBound42744
def owner : Owner := ⟨.program ⟨214⟩, ⟨27242⟩⟩
def transferEvent : Nat := 42744
def frameStart : Nat := 42667
def rule : BoundRule := .product (.predecessor 0 42742 .coefficient) (.predecessor 1 42743 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42742 .coefficient)
      LeftBound42740.bound (LeftBound42740.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events166.exact42741RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42740.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42740.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42743 .coefficient)
      LeftAuthority42717.bound (LeftAuthority42717.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events166.exact42718RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42717.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42717.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound42740.bound LeftAuthority42717.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42740.bound, LeftAuthority42717.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound42740.actual selector witness) * (LeftAuthority42717.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound42744

namespace LeftBound42755
def owner : Owner := ⟨.program ⟨214⟩, ⟨15636⟩⟩
def transferEvent : Nat := 42755
def frameStart : Nat := 42667
def rule : BoundRule := .product (.predecessor 0 42753 .coefficient) (.predecessor 1 42754 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42753 .coefficient)
      LeftAuthority42728.bound (LeftAuthority42728.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events166.exact42729RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42728.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42728.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42754 .coefficient)
      LeftAuthority42751.bound (LeftAuthority42751.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events167.exact42752RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42751.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42751.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority42728.bound LeftAuthority42751.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority42728.bound, LeftAuthority42751.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority42728.actual selector witness) * (LeftAuthority42751.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound42755

namespace LeftBound42763
def owner : Owner := ⟨.program ⟨214⟩, ⟨15637⟩⟩
def transferEvent : Nat := 42763
def frameStart : Nat := 42667
def rule : BoundRule := .sum [.predecessor 0 42761 .coefficient, .predecessor 1 42762 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42761 .coefficient)
      LeftAuthority42759.bound (LeftAuthority42759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events167.exact42760RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42759.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42762 .coefficient)
      LeftBound42755.bound (LeftBound42755.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events167.exact42757RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42755.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42755.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority42759.bound, LeftBound42755.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority42759.bound, LeftBound42755.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority42759.actual selector witness, LeftBound42755.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound42763

namespace LeftBound42767
def owner : Owner := ⟨.program ⟨214⟩, ⟨27246⟩⟩
def transferEvent : Nat := 42767
def frameStart : Nat := 42667
def rule : BoundRule := .sum [.predecessor 0 42765 .coefficient, .predecessor 1 42766 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42765 .coefficient)
      LeftBound42763.bound (LeftBound42763.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events167.exact42764RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42763.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42763.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42766 .coefficient)
      LeftBound42744.bound (LeftBound42744.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events166.exact42749RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42744.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42744.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound42763.bound, LeftBound42744.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42763.bound, LeftBound42744.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound42763.actual selector witness, LeftBound42744.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound42767

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
