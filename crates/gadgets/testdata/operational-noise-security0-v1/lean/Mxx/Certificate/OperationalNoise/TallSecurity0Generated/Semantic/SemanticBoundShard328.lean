import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard294
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard327

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound49534
def owner : Owner := ⟨.program ⟨214⟩, ⟨26802⟩⟩
def transferEvent : Nat := 49534
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 43553 .summary) (.transfer 49533) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 43553 .summary)
      LeftBound43552.bound (LeftBound43552.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25077⟩⟩) (rawTerms := some (Proof.Events170.exact43553RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound43552.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 49533)
      LeftBound49533.bound (LeftBound49533.actual selector witness) := by
  exact .transfer (LeftBound49533.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound43552.bound LeftBound49533.bound
def bound : CoeffClass := .finite ⟨1291911585013138718720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43552.bound, LeftBound49533.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound43552.actual selector witness) * (LeftBound49533.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound49534

namespace LeftBound49545
def owner : Owner := ⟨.program ⟨214⟩, ⟨20618⟩⟩
def transferEvent : Nat := 49545
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 49543 .coefficient) (.value (.predecessor 1 49544 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49543 .coefficient)
      LeftAuthority49541.bound (LeftAuthority49541.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events193.exact49542RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49541.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49541.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49544 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority49541.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority49541.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority49541.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound49545

namespace LeftBound49549
def owner : Owner := ⟨.program ⟨214⟩, ⟨20619⟩⟩
def transferEvent : Nat := 49549
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 49547 .coefficient) (.predecessor 1 49548 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49547 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49548 .coefficient)
      LeftBound49545.bound (LeftBound49545.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events193.exact49546RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49545.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49545.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound49545.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound49545.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound49545.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound49549

namespace LeftBound49550
def owner : Owner := ⟨.program ⟨214⟩, ⟨20619⟩⟩
def transferEvent : Nat := 49550
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20616⟩⟩]⟩ [⟨.result 49542 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 49542 .coefficient)
      LeftAuthority49541.bound (LeftAuthority49541.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20616⟩⟩) (rawTerms := some (Proof.Events193.exact49542RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49541.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49541.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority49541.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority49541.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority49541.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound49550

namespace LeftBound49551
def owner : Owner := ⟨.program ⟨214⟩, ⟨20619⟩⟩
def transferEvent : Nat := 49551
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 49550) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 49550)
      LeftBound49550.bound (LeftBound49550.actual selector witness) := by
  exact .transfer (LeftBound49550.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound49550.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound49550.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound49550.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound49551

namespace LeftBound49646
def owner : Owner := ⟨.program ⟨214⟩, ⟨15123⟩⟩
def transferEvent : Nat := 49646
def frameStart : Nat := 49607
def rule : BoundRule := .identity (.predecessor 0 49645 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49645 .coefficient)
      LeftAuthority49643.bound (LeftAuthority49643.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events193.exact49644RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49643.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49643.derived selector witness)

def rawBound : CoeffClass := LeftAuthority49643.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority49643.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority49643.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound49646

namespace LeftBound49663
def owner : Owner := ⟨.program ⟨214⟩, ⟨15162⟩⟩
def transferEvent : Nat := 49663
def frameStart : Nat := 49607
def rule : BoundRule := .sum [.predecessor 0 49661 .coefficient, .predecessor 1 49662 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49661 .coefficient)
      LeftBound49646.bound (LeftBound49646.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound49646.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49662 .coefficient)
      LeftAuthority49659.bound (LeftAuthority49659.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority49659.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound49646.bound, LeftAuthority49659.bound]
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound49646.bound, LeftAuthority49659.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound49646.actual selector witness, LeftAuthority49659.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound49663

namespace LeftBound49666
def owner : Owner := ⟨.program ⟨214⟩, ⟨15163⟩⟩
def transferEvent : Nat := 49666
def frameStart : Nat := 49607
def rule : BoundRule := .identity (.predecessor 0 49665 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49665 .coefficient)
      LeftBound49663.bound (LeftBound49663.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound49663.derived selector witness)

def rawBound : CoeffClass := LeftBound49663.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound49663.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound49663.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound49666

namespace LeftBound49672
def owner : Owner := ⟨.program ⟨214⟩, ⟨15164⟩⟩
def transferEvent : Nat := 49672
def frameStart : Nat := 49607
def rule : BoundRule := .product (.predecessor 0 49670 .coefficient) (.predecessor 1 49671 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49670 .coefficient)
      LeftAuthority49668.bound (LeftAuthority49668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events194.exact49669RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49668.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49668.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49671 .coefficient)
      LeftBound49666.bound (LeftBound49666.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events194.exact49667RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49666.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49666.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority49668.bound LeftBound49666.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority49668.bound, LeftBound49666.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority49668.actual selector witness) * (LeftBound49666.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound49672

namespace LeftBound49680
def owner : Owner := ⟨.program ⟨214⟩, ⟨15165⟩⟩
def transferEvent : Nat := 49680
def frameStart : Nat := 49607
def rule : BoundRule := .sum [.predecessor 0 49678 .coefficient, .predecessor 1 49679 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49678 .coefficient)
      LeftAuthority49676.bound (LeftAuthority49676.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events194.exact49677RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49676.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49676.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49679 .coefficient)
      LeftBound49672.bound (LeftBound49672.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events194.exact49674RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49672.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49672.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority49676.bound, LeftBound49672.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority49676.bound, LeftBound49672.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority49676.actual selector witness, LeftBound49672.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound49680

namespace LeftBound49684
def owner : Owner := ⟨.program ⟨214⟩, ⟨26801⟩⟩
def transferEvent : Nat := 49684
def frameStart : Nat := 49607
def rule : BoundRule := .product (.predecessor 0 49682 .coefficient) (.predecessor 1 49683 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49682 .coefficient)
      LeftBound49680.bound (LeftBound49680.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events194.exact49681RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49680.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49680.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49683 .coefficient)
      LeftAuthority49657.bound (LeftAuthority49657.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events193.exact49658RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49657.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49657.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound49680.bound LeftAuthority49657.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound49680.bound, LeftAuthority49657.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound49680.actual selector witness) * (LeftAuthority49657.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound49684

namespace LeftBound49695
def owner : Owner := ⟨.program ⟨214⟩, ⟨15221⟩⟩
def transferEvent : Nat := 49695
def frameStart : Nat := 49607
def rule : BoundRule := .product (.predecessor 0 49693 .coefficient) (.predecessor 1 49694 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49693 .coefficient)
      LeftAuthority49668.bound (LeftAuthority49668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events194.exact49669RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49668.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49668.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49694 .coefficient)
      LeftAuthority49691.bound (LeftAuthority49691.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events194.exact49692RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49691.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49691.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority49668.bound LeftAuthority49691.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority49668.bound, LeftAuthority49691.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority49668.actual selector witness) * (LeftAuthority49691.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound49695

namespace LeftBound49703
def owner : Owner := ⟨.program ⟨214⟩, ⟨15222⟩⟩
def transferEvent : Nat := 49703
def frameStart : Nat := 49607
def rule : BoundRule := .sum [.predecessor 0 49701 .coefficient, .predecessor 1 49702 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49701 .coefficient)
      LeftAuthority49699.bound (LeftAuthority49699.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events194.exact49700RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49699.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49699.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49702 .coefficient)
      LeftBound49695.bound (LeftBound49695.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events194.exact49697RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49695.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49695.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority49699.bound, LeftBound49695.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority49699.bound, LeftBound49695.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority49699.actual selector witness, LeftBound49695.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound49703

namespace LeftBound49707
def owner : Owner := ⟨.program ⟨214⟩, ⟨26806⟩⟩
def transferEvent : Nat := 49707
def frameStart : Nat := 49607
def rule : BoundRule := .sum [.predecessor 0 49705 .coefficient, .predecessor 1 49706 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49705 .coefficient)
      LeftBound49703.bound (LeftBound49703.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events194.exact49704RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49703.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49703.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49706 .coefficient)
      LeftBound49684.bound (LeftBound49684.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events194.exact49689RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49684.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49684.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound49703.bound, LeftBound49684.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound49703.bound, LeftBound49684.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound49703.actual selector witness, LeftBound49684.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound49707

namespace LeftBound49720
def owner : Owner := ⟨.program ⟨214⟩, ⟨26803⟩⟩
def transferEvent : Nat := 49720
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 49718 .coefficient, .predecessor 1 49719 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49718 .coefficient)
      LeftBound49549.bound (LeftBound49549.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events194.exact49717RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49549.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49549.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49719 .coefficient)
      LeftBound49532.bound (LeftBound49532.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events193.exact49539RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49532.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49532.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound49549.bound, LeftBound49532.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound49549.bound, LeftBound49532.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound49549.actual selector witness, LeftBound49532.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound49720

namespace LeftBound49723
def owner : Owner := ⟨.program ⟨214⟩, ⟨26803⟩⟩
def transferEvent : Nat := 49723
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 49717 .summary, .result 49539 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 49717 .summary)
      LeftBound49551.bound (LeftBound49551.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20619⟩⟩) (rawTerms := some (Proof.Events194.exact49717RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound49551.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 49539 .summary)
      LeftBound49534.bound (LeftBound49534.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26802⟩⟩) (rawTerms := some (Proof.Events193.exact49539RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound49534.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound49551.bound, LeftBound49534.bound]
def bound : CoeffClass := .finite ⟨1291911586824442228736, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound49551.bound, LeftBound49534.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound49551.actual selector witness, LeftBound49534.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound49723

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
