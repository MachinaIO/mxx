import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard019
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard478
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard523

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound77705
def owner : Owner := ⟨.program ⟨214⟩, ⟨28067⟩⟩
def transferEvent : Nat := 77705
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 77703 .coefficient) (.predecessor 1 77704 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77703 .coefficient)
      LeftBound77698.bound (LeftBound77698.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events303.exact77702RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77698.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77698.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77704 .coefficient)
      LeftBound5698.bound (LeftBound5698.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5699RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5698.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5698.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound77698.bound LeftBound5698.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77698.bound, LeftBound5698.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound77698.actual selector witness) * (LeftBound5698.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77705

namespace LeftBound77706
def owner : Owner := ⟨.program ⟨214⟩, ⟨28067⟩⟩
def transferEvent : Nat := 77706
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6637⟩⟩]⟩ [⟨.result 5695 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5695 .coefficient)
      LeftAuthority5694.bound (LeftAuthority5694.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6637⟩⟩) (rawTerms := some (Proof.Events022.exact5695RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5694.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5694.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5694.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5694.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5694.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound77706

namespace LeftBound77707
def owner : Owner := ⟨.program ⟨214⟩, ⟨28067⟩⟩
def transferEvent : Nat := 77707
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 77702 .summary) (.transfer 77706) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 77702 .summary)
      LeftBound77701.bound (LeftBound77701.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28066⟩⟩) (rawTerms := some (Proof.Events303.exact77702RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound77701.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 77706)
      LeftBound77706.bound (LeftBound77706.actual selector witness) := by
  exact .transfer (LeftBound77706.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound77701.bound LeftBound77706.bound
def bound : CoeffClass := .finite ⟨4742076480517514208552681472, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77701.bound, LeftBound77706.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound77701.actual selector witness) * (LeftBound77706.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77707

namespace LeftBound77722
def owner : Owner := ⟨.program ⟨214⟩, ⟨27848⟩⟩
def transferEvent : Nat := 77722
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 77720 .coefficient) (.predecessor 1 77721 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77720 .coefficient)
      LeftBound70389.bound (LeftBound70389.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events274.exact70393RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70389.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70389.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77721 .coefficient)
      LeftAuthority77718.bound (LeftAuthority77718.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events303.exact77719RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77718.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77718.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound70389.bound LeftAuthority77718.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70389.bound, LeftAuthority77718.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound70389.actual selector witness) * (LeftAuthority77718.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77722

namespace LeftBound77723
def owner : Owner := ⟨.program ⟨214⟩, ⟨27848⟩⟩
def transferEvent : Nat := 77723
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27846⟩⟩]⟩ [⟨.result 77719 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 77719 .coefficient)
      LeftAuthority77718.bound (LeftAuthority77718.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27846⟩⟩) (rawTerms := some (Proof.Events303.exact77719RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77718.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77718.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority77718.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority77718.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority77718.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound77723

namespace LeftBound77724
def owner : Owner := ⟨.program ⟨214⟩, ⟨27848⟩⟩
def transferEvent : Nat := 77724
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 70393 .summary) (.transfer 77723) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 70393 .summary)
      LeftBound70392.bound (LeftBound70392.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26063⟩⟩) (rawTerms := some (Proof.Events274.exact70393RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound70392.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 77723)
      LeftBound77723.bound (LeftBound77723.actual selector witness) := by
  exact .transfer (LeftBound77723.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound70392.bound LeftBound77723.bound
def bound : CoeffClass := .finite ⟨1292068472128282820608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70392.bound, LeftBound77723.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound70392.actual selector witness) * (LeftBound77723.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77724

namespace LeftBound77735
def owner : Owner := ⟨.program ⟨214⟩, ⟨21326⟩⟩
def transferEvent : Nat := 77735
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 77733 .coefficient) (.value (.predecessor 1 77734 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77733 .coefficient)
      LeftAuthority77731.bound (LeftAuthority77731.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events303.exact77732RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77731.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77731.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77734 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority77731.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority77731.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority77731.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound77735

namespace LeftBound77739
def owner : Owner := ⟨.program ⟨214⟩, ⟨21327⟩⟩
def transferEvent : Nat := 77739
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 77737 .coefficient) (.predecessor 1 77738 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77737 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77738 .coefficient)
      LeftBound77735.bound (LeftBound77735.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events303.exact77736RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77735.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77735.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound77735.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound77735.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound77735.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77739

namespace LeftBound77740
def owner : Owner := ⟨.program ⟨214⟩, ⟨21327⟩⟩
def transferEvent : Nat := 77740
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21324⟩⟩]⟩ [⟨.result 77732 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 77732 .coefficient)
      LeftAuthority77731.bound (LeftAuthority77731.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21324⟩⟩) (rawTerms := some (Proof.Events303.exact77732RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77731.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77731.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority77731.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority77731.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority77731.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound77740

namespace LeftBound77741
def owner : Owner := ⟨.program ⟨214⟩, ⟨21327⟩⟩
def transferEvent : Nat := 77741
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 77740) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 77740)
      LeftBound77740.bound (LeftBound77740.actual selector witness) := by
  exact .transfer (LeftBound77740.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound77740.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound77740.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound77740.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77741

namespace LeftBound77836
def owner : Owner := ⟨.program ⟨214⟩, ⟨15937⟩⟩
def transferEvent : Nat := 77836
def frameStart : Nat := 77797
def rule : BoundRule := .identity (.predecessor 0 77835 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77835 .coefficient)
      LeftAuthority77833.bound (LeftAuthority77833.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events304.exact77834RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77833.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77833.derived selector witness)

def rawBound : CoeffClass := LeftAuthority77833.bound
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority77833.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority77833.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound77836

namespace LeftBound77853
def owner : Owner := ⟨.program ⟨214⟩, ⟨16011⟩⟩
def transferEvent : Nat := 77853
def frameStart : Nat := 77797
def rule : BoundRule := .sum [.predecessor 0 77851 .coefficient, .predecessor 1 77852 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77851 .coefficient)
      LeftBound77836.bound (LeftBound77836.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound77836.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77852 .coefficient)
      LeftAuthority77849.bound (LeftAuthority77849.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority77849.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound77836.bound, LeftAuthority77849.bound]
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77836.bound, LeftAuthority77849.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound77836.actual selector witness, LeftAuthority77849.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound77853

namespace LeftBound77856
def owner : Owner := ⟨.program ⟨214⟩, ⟨16012⟩⟩
def transferEvent : Nat := 77856
def frameStart : Nat := 77797
def rule : BoundRule := .identity (.predecessor 0 77855 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77855 .coefficient)
      LeftBound77853.bound (LeftBound77853.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound77853.derived selector witness)

def rawBound : CoeffClass := LeftBound77853.bound
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77853.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound77853.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound77856

namespace LeftBound77862
def owner : Owner := ⟨.program ⟨214⟩, ⟨16013⟩⟩
def transferEvent : Nat := 77862
def frameStart : Nat := 77797
def rule : BoundRule := .product (.predecessor 0 77860 .coefficient) (.predecessor 1 77861 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77860 .coefficient)
      LeftAuthority77858.bound (LeftAuthority77858.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events304.exact77859RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77858.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77858.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77861 .coefficient)
      LeftBound77856.bound (LeftBound77856.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events304.exact77857RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77856.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77856.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority77858.bound LeftBound77856.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority77858.bound, LeftBound77856.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority77858.actual selector witness) * (LeftBound77856.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77862

namespace LeftBound77870
def owner : Owner := ⟨.program ⟨214⟩, ⟨16014⟩⟩
def transferEvent : Nat := 77870
def frameStart : Nat := 77797
def rule : BoundRule := .sum [.predecessor 0 77868 .coefficient, .predecessor 1 77869 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77868 .coefficient)
      LeftAuthority77866.bound (LeftAuthority77866.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events304.exact77867RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77866.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77866.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77869 .coefficient)
      LeftBound77862.bound (LeftBound77862.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events304.exact77864RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77862.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77862.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority77866.bound, LeftBound77862.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority77866.bound, LeftBound77862.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority77866.actual selector witness, LeftBound77862.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound77870

namespace LeftBound77874
def owner : Owner := ⟨.program ⟨214⟩, ⟨27847⟩⟩
def transferEvent : Nat := 77874
def frameStart : Nat := 77797
def rule : BoundRule := .product (.predecessor 0 77872 .coefficient) (.predecessor 1 77873 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77872 .coefficient)
      LeftBound77870.bound (LeftBound77870.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events304.exact77871RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77870.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77870.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77873 .coefficient)
      LeftAuthority77847.bound (LeftAuthority77847.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events304.exact77848RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77847.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77847.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound77870.bound LeftAuthority77847.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77870.bound, LeftAuthority77847.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound77870.actual selector witness) * (LeftAuthority77847.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77874

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
