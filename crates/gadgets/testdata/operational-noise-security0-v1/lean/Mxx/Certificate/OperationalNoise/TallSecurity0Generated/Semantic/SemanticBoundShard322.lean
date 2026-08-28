import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard279
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard321

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound48620
def owner : Owner := ⟨.program ⟨214⟩, ⟨16026⟩⟩
def transferEvent : Nat := 48620
def frameStart : Nat := 48547
def rule : BoundRule := .sum [.predecessor 0 48618 .coefficient, .predecessor 1 48619 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48618 .coefficient)
      LeftAuthority48616.bound (LeftAuthority48616.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48617RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48616.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48616.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48619 .coefficient)
      LeftBound48612.bound (LeftBound48612.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48614RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48612.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48612.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority48616.bound, LeftBound48612.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48616.bound, LeftBound48612.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority48616.actual selector witness, LeftBound48612.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound48620

namespace LeftBound48624
def owner : Owner := ⟨.program ⟨214⟩, ⟨27886⟩⟩
def transferEvent : Nat := 48624
def frameStart : Nat := 48547
def rule : BoundRule := .product (.predecessor 0 48622 .coefficient) (.predecessor 1 48623 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48622 .coefficient)
      LeftBound48620.bound (LeftBound48620.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48621RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48620.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48620.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48623 .coefficient)
      LeftAuthority48597.bound (LeftAuthority48597.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48598RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48597.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48597.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound48620.bound LeftAuthority48597.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48620.bound, LeftAuthority48597.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound48620.actual selector witness) * (LeftAuthority48597.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48624

namespace LeftBound48635
def owner : Owner := ⟨.program ⟨214⟩, ⟨17175⟩⟩
def transferEvent : Nat := 48635
def frameStart : Nat := 48547
def rule : BoundRule := .product (.predecessor 0 48633 .coefficient) (.predecessor 1 48634 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48633 .coefficient)
      LeftAuthority48608.bound (LeftAuthority48608.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48609RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48608.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48608.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48634 .coefficient)
      LeftAuthority48631.bound (LeftAuthority48631.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48632RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48631.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48631.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority48608.bound LeftAuthority48631.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48608.bound, LeftAuthority48631.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority48608.actual selector witness) * (LeftAuthority48631.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48635

namespace LeftBound48643
def owner : Owner := ⟨.program ⟨214⟩, ⟨17176⟩⟩
def transferEvent : Nat := 48643
def frameStart : Nat := 48547
def rule : BoundRule := .sum [.predecessor 0 48641 .coefficient, .predecessor 1 48642 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48641 .coefficient)
      LeftAuthority48639.bound (LeftAuthority48639.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events190.exact48640RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48639.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48639.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48642 .coefficient)
      LeftBound48635.bound (LeftBound48635.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48637RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48635.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48635.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority48639.bound, LeftBound48635.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48639.bound, LeftBound48635.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority48639.actual selector witness, LeftBound48635.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound48643

namespace LeftBound48647
def owner : Owner := ⟨.program ⟨214⟩, ⟨27891⟩⟩
def transferEvent : Nat := 48647
def frameStart : Nat := 48547
def rule : BoundRule := .sum [.predecessor 0 48645 .coefficient, .predecessor 1 48646 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48645 .coefficient)
      LeftBound48643.bound (LeftBound48643.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events190.exact48644RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48643.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48643.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48646 .coefficient)
      LeftBound48624.bound (LeftBound48624.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48629RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48624.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48624.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound48643.bound, LeftBound48624.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48643.bound, LeftBound48624.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound48643.actual selector witness, LeftBound48624.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound48647

namespace LeftBound48660
def owner : Owner := ⟨.program ⟨214⟩, ⟨27888⟩⟩
def transferEvent : Nat := 48660
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 48658 .coefficient, .predecessor 1 48659 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48658 .coefficient)
      LeftBound48489.bound (LeftBound48489.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events190.exact48657RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48489.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48489.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48659 .coefficient)
      LeftBound48472.bound (LeftBound48472.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48479RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48472.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48472.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound48489.bound, LeftBound48472.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48489.bound, LeftBound48472.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound48489.actual selector witness, LeftBound48472.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound48660

namespace LeftBound48663
def owner : Owner := ⟨.program ⟨214⟩, ⟨27888⟩⟩
def transferEvent : Nat := 48663
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 48657 .summary, .result 48479 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 48657 .summary)
      LeftBound48491.bound (LeftBound48491.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21339⟩⟩) (rawTerms := some (Proof.Events190.exact48657RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound48491.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 48479 .summary)
      LeftBound48474.bound (LeftBound48474.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27887⟩⟩) (rawTerms := some (Proof.Events189.exact48479RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound48474.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound48491.bound, LeftBound48474.bound]
def bound : CoeffClass := .finite ⟨1292068473939586330624, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48491.bound, LeftBound48474.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound48491.actual selector witness, LeftBound48474.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound48663

namespace LeftBound48667
def owner : Owner := ⟨.program ⟨214⟩, ⟨27889⟩⟩
def transferEvent : Nat := 48667
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 48665 .coefficient) (.predecessor 1 48666 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48665 .coefficient)
      LeftBound48660.bound (LeftBound48660.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events190.exact48664RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48660.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48660.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48666 .coefficient)
      LeftBound5718.bound (LeftBound5718.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5719RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5718.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5718.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound48660.bound LeftBound5718.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48660.bound, LeftBound5718.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound48660.actual selector witness) * (LeftBound5718.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48667

namespace LeftBound48668
def owner : Owner := ⟨.program ⟨214⟩, ⟨27889⟩⟩
def transferEvent : Nat := 48668
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6641⟩⟩]⟩ [⟨.result 5715 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5715 .coefficient)
      LeftAuthority5714.bound (LeftAuthority5714.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6641⟩⟩) (rawTerms := some (Proof.Events022.exact5715RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5714.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5714.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5714.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5714.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5714.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound48668

namespace LeftBound48669
def owner : Owner := ⟨.program ⟨214⟩, ⟨27889⟩⟩
def transferEvent : Nat := 48669
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 48664 .summary) (.transfer 48668) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 48664 .summary)
      LeftBound48663.bound (LeftBound48663.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27888⟩⟩) (rawTerms := some (Proof.Events190.exact48664RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound48663.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 48668)
      LeftBound48668.bound (LeftBound48668.actual selector witness) := by
  exact .transfer (LeftBound48668.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound48663.bound LeftBound48668.bound
def bound : CoeffClass := .finite ⟨4741911972453864866771369984, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48663.bound, LeftBound48668.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound48663.actual selector witness) * (LeftBound48668.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48669

namespace LeftBound48684
def owner : Owner := ⟨.program ⟨214⟩, ⟨27670⟩⟩
def transferEvent : Nat := 48684
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 48682 .coefficient) (.predecessor 1 48683 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48682 .coefficient)
      LeftBound41621.bound (LeftBound41621.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events162.exact41625RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41621.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41621.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48683 .coefficient)
      LeftAuthority48680.bound (LeftAuthority48680.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events190.exact48681RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48680.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48680.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound41621.bound LeftAuthority48680.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41621.bound, LeftAuthority48680.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound41621.actual selector witness) * (LeftAuthority48680.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48684

namespace LeftBound48685
def owner : Owner := ⟨.program ⟨214⟩, ⟨27670⟩⟩
def transferEvent : Nat := 48685
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27668⟩⟩]⟩ [⟨.result 48681 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 48681 .coefficient)
      LeftAuthority48680.bound (LeftAuthority48680.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27668⟩⟩) (rawTerms := some (Proof.Events190.exact48681RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48680.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48680.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority48680.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48680.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority48680.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound48685

namespace LeftBound48686
def owner : Owner := ⟨.program ⟨214⟩, ⟨27670⟩⟩
def transferEvent : Nat := 48686
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 41625 .summary) (.transfer 48685) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 41625 .summary)
      LeftBound41624.bound (LeftBound41624.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26001⟩⟩) (rawTerms := some (Proof.Events162.exact41625RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound41624.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 48685)
      LeftBound48685.bound (LeftBound48685.actual selector witness) := by
  exact .transfer (LeftBound48685.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound41624.bound LeftBound48685.bound
def bound : CoeffClass := .finite ⟨1292046059683262234624, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41624.bound, LeftBound48685.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound41624.actual selector witness) * (LeftBound48685.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48686

namespace LeftBound48697
def owner : Owner := ⟨.program ⟨214⟩, ⟨21194⟩⟩
def transferEvent : Nat := 48697
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 48695 .coefficient) (.value (.predecessor 1 48696 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48695 .coefficient)
      LeftAuthority48693.bound (LeftAuthority48693.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events190.exact48694RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48693.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48693.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48696 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority48693.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48693.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority48693.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound48697

namespace LeftBound48701
def owner : Owner := ⟨.program ⟨214⟩, ⟨21195⟩⟩
def transferEvent : Nat := 48701
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 48699 .coefficient) (.predecessor 1 48700 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48699 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48700 .coefficient)
      LeftBound48697.bound (LeftBound48697.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events190.exact48698RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48697.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48697.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound48697.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound48697.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound48697.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48701

namespace LeftBound48702
def owner : Owner := ⟨.program ⟨214⟩, ⟨21195⟩⟩
def transferEvent : Nat := 48702
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21192⟩⟩]⟩ [⟨.result 48694 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 48694 .coefficient)
      LeftAuthority48693.bound (LeftAuthority48693.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21192⟩⟩) (rawTerms := some (Proof.Events190.exact48694RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48693.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48693.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority48693.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48693.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority48693.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound48702

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
