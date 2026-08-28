import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard032
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard437
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard440
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard442

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound65593
def owner : Owner := ⟨.program ⟨214⟩, ⟨22839⟩⟩
def transferEvent : Nat := 65593
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 65591 .coefficient) (.predecessor 1 65592 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65591 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 65592 .coefficient)
      LeftBound65589.bound (LeftBound65589.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65590RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65589.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65589.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound65589.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound65589.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound65589.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound65593

namespace LeftBound65594
def owner : Owner := ⟨.program ⟨214⟩, ⟨22839⟩⟩
def transferEvent : Nat := 65594
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22836⟩⟩]⟩ [⟨.result 65586 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65586 .coefficient)
      LeftAuthority65585.bound (LeftAuthority65585.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22836⟩⟩) (rawTerms := some (Proof.Events256.exact65586RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65585.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65585.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority65585.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority65585.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority65585.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound65594

namespace LeftBound65595
def owner : Owner := ⟨.program ⟨214⟩, ⟨22839⟩⟩
def transferEvent : Nat := 65595
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 65594) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 65594)
      LeftBound65594.bound (LeftBound65594.actual selector witness) := by
  exact .transfer (LeftBound65594.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound65594.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound65594.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound65594.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound65595

namespace LeftBound65690
def owner : Owner := ⟨.program ⟨214⟩, ⟨17008⟩⟩
def transferEvent : Nat := 65690
def frameStart : Nat := 65651
def rule : BoundRule := .identity (.predecessor 0 65689 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65689 .coefficient)
      LeftAuthority65687.bound (LeftAuthority65687.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65688RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65687.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65687.derived selector witness)

def rawBound : CoeffClass := LeftAuthority65687.bound
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority65687.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority65687.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound65690

namespace LeftBound65707
def owner : Owner := ⟨.program ⟨214⟩, ⟨17047⟩⟩
def transferEvent : Nat := 65707
def frameStart : Nat := 65651
def rule : BoundRule := .sum [.predecessor 0 65705 .coefficient, .predecessor 1 65706 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65705 .coefficient)
      LeftBound65690.bound (LeftBound65690.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound65690.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 65706 .coefficient)
      LeftAuthority65703.bound (LeftAuthority65703.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority65703.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound65690.bound, LeftAuthority65703.bound]
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65690.bound, LeftAuthority65703.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound65690.actual selector witness, LeftAuthority65703.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound65707

namespace LeftBound65710
def owner : Owner := ⟨.program ⟨214⟩, ⟨17048⟩⟩
def transferEvent : Nat := 65710
def frameStart : Nat := 65651
def rule : BoundRule := .identity (.predecessor 0 65709 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65709 .coefficient)
      LeftBound65707.bound (LeftBound65707.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound65707.derived selector witness)

def rawBound : CoeffClass := LeftBound65707.bound
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65707.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound65707.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound65710

namespace LeftBound65716
def owner : Owner := ⟨.program ⟨214⟩, ⟨17049⟩⟩
def transferEvent : Nat := 65716
def frameStart : Nat := 65651
def rule : BoundRule := .product (.predecessor 0 65714 .coefficient) (.predecessor 1 65715 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65714 .coefficient)
      LeftAuthority65712.bound (LeftAuthority65712.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65713RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65712.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65712.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 65715 .coefficient)
      LeftBound65710.bound (LeftBound65710.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65711RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65710.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65710.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority65712.bound LeftBound65710.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority65712.bound, LeftBound65710.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority65712.actual selector witness) * (LeftBound65710.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound65716

namespace LeftBound65724
def owner : Owner := ⟨.program ⟨214⟩, ⟨17050⟩⟩
def transferEvent : Nat := 65724
def frameStart : Nat := 65651
def rule : BoundRule := .sum [.predecessor 0 65722 .coefficient, .predecessor 1 65723 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65722 .coefficient)
      LeftAuthority65720.bound (LeftAuthority65720.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65721RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65720.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65720.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 65723 .coefficient)
      LeftBound65716.bound (LeftBound65716.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65718RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65716.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65716.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority65720.bound, LeftBound65716.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority65720.bound, LeftBound65716.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority65720.actual selector witness, LeftBound65716.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound65724

namespace LeftBound65728
def owner : Owner := ⟨.program ⟨214⟩, ⟨30096⟩⟩
def transferEvent : Nat := 65728
def frameStart : Nat := 65651
def rule : BoundRule := .product (.predecessor 0 65726 .coefficient) (.predecessor 1 65727 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65726 .coefficient)
      LeftBound65724.bound (LeftBound65724.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65725RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65724.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65724.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 65727 .coefficient)
      LeftAuthority65701.bound (LeftAuthority65701.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65702RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65701.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65701.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound65724.bound LeftAuthority65701.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65724.bound, LeftAuthority65701.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound65724.actual selector witness) * (LeftAuthority65701.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound65728

namespace LeftBound65739
def owner : Owner := ⟨.program ⟨214⟩, ⟨18168⟩⟩
def transferEvent : Nat := 65739
def frameStart : Nat := 65651
def rule : BoundRule := .product (.predecessor 0 65737 .coefficient) (.predecessor 1 65738 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65737 .coefficient)
      LeftAuthority65712.bound (LeftAuthority65712.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65713RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65712.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65712.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 65738 .coefficient)
      LeftAuthority65735.bound (LeftAuthority65735.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65736RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65735.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65735.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority65712.bound LeftAuthority65735.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority65712.bound, LeftAuthority65735.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority65712.actual selector witness) * (LeftAuthority65735.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound65739

namespace LeftBound65747
def owner : Owner := ⟨.program ⟨214⟩, ⟨18169⟩⟩
def transferEvent : Nat := 65747
def frameStart : Nat := 65651
def rule : BoundRule := .sum [.predecessor 0 65745 .coefficient, .predecessor 1 65746 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65745 .coefficient)
      LeftAuthority65743.bound (LeftAuthority65743.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65744RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65743.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65743.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 65746 .coefficient)
      LeftBound65739.bound (LeftBound65739.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65741RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65739.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65739.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority65743.bound, LeftBound65739.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority65743.bound, LeftBound65739.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority65743.actual selector witness, LeftBound65739.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound65747

namespace LeftBound65751
def owner : Owner := ⟨.program ⟨214⟩, ⟨30103⟩⟩
def transferEvent : Nat := 65751
def frameStart : Nat := 65651
def rule : BoundRule := .sum [.predecessor 0 65749 .coefficient, .predecessor 1 65750 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65749 .coefficient)
      LeftBound65747.bound (LeftBound65747.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65748RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65747.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65747.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 65750 .coefficient)
      LeftBound65728.bound (LeftBound65728.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65733RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65728.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65728.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound65747.bound, LeftBound65728.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65747.bound, LeftBound65728.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound65747.actual selector witness, LeftBound65728.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound65751

namespace LeftBound65764
def owner : Owner := ⟨.program ⟨214⟩, ⟨30098⟩⟩
def transferEvent : Nat := 65764
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 65762 .coefficient, .predecessor 1 65763 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65762 .coefficient)
      LeftBound65593.bound (LeftBound65593.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65761RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65593.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65593.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 65763 .coefficient)
      LeftBound65576.bound (LeftBound65576.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65583RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65576.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65576.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound65593.bound, LeftBound65576.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65593.bound, LeftBound65576.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound65593.actual selector witness, LeftBound65576.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound65764

namespace LeftBound65767
def owner : Owner := ⟨.program ⟨214⟩, ⟨30098⟩⟩
def transferEvent : Nat := 65767
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 65761 .summary, .result 65583 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65761 .summary)
      LeftBound65595.bound (LeftBound65595.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22839⟩⟩) (rawTerms := some (Proof.Events256.exact65761RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65595.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65583 .summary)
      LeftBound65578.bound (LeftBound65578.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30097⟩⟩) (rawTerms := some (Proof.Events256.exact65583RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65578.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound65595.bound, LeftBound65578.bound]
def bound : CoeffClass := .finite ⟨1292539135285018636288, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65595.bound, LeftBound65578.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound65595.actual selector witness, LeftBound65578.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound65767

namespace LeftBound65791
def owner : Owner := ⟨.program ⟨214⟩, ⟨13149⟩⟩
def transferEvent : Nat := 65791
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 65789 .coefficient) (.predecessor 1 65790 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65789 .coefficient)
      LeftAuthority3108.bound (LeftAuthority3108.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events012.exact3109RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3108.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3108.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 65790 .coefficient)
      LeftBound65293.bound (LeftBound65293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65293.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority3108.bound LeftBound65293.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3108.bound, LeftBound65293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority3108.actual selector witness) * (LeftBound65293.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound65791

namespace LeftBound65796
def owner : Owner := ⟨.program ⟨214⟩, ⟨7207⟩⟩
def transferEvent : Nat := 65796
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 65794 .coefficient) (.predecessor 1 65795 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65794 .coefficient)
      LeftBound65164.bound (LeftBound65164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65164.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 65795 .coefficient)
      LeftBound6972.bound (LeftBound6972.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact6973RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6972.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6972.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound65164.bound LeftBound6972.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65164.bound, LeftBound6972.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound65164.actual selector witness) * (LeftBound6972.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound65796

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
