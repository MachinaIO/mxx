import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard072
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard437
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard440
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard479

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound70584
def owner : Owner := ⟨.program ⟨214⟩, ⟨27856⟩⟩
def transferEvent : Nat := 70584
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 70582 .coefficient, .predecessor 1 70583 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70582 .coefficient)
      LeftBound70413.bound (LeftBound70413.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events275.exact70581RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70413.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70413.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70583 .coefficient)
      LeftBound70396.bound (LeftBound70396.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events275.exact70403RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70396.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70396.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound70413.bound, LeftBound70396.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70413.bound, LeftBound70396.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound70413.actual selector witness, LeftBound70396.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound70584

namespace LeftBound70587
def owner : Owner := ⟨.program ⟨214⟩, ⟨27856⟩⟩
def transferEvent : Nat := 70587
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 70581 .summary, .result 70403 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 70581 .summary)
      LeftBound70415.bound (LeftBound70415.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21399⟩⟩) (rawTerms := some (Proof.Events275.exact70581RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound70415.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 70403 .summary)
      LeftBound70398.bound (LeftBound70398.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27855⟩⟩) (rawTerms := some (Proof.Events275.exact70403RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound70398.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound70415.bound, LeftBound70398.bound]
def bound : CoeffClass := .finite ⟨1292068473939586330624, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70415.bound, LeftBound70398.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound70415.actual selector witness, LeftBound70398.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound70587

namespace LeftBound70611
def owner : Owner := ⟨.program ⟨214⟩, ⟨11382⟩⟩
def transferEvent : Nat := 70611
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 70609 .coefficient) (.predecessor 1 70610 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70609 .coefficient)
      LeftAuthority3338.bound (LeftAuthority3338.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events013.exact3339RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3338.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3338.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70610 .coefficient)
      LeftBound65293.bound (LeftBound65293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65293.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority3338.bound LeftBound65293.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3338.bound, LeftBound65293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority3338.actual selector witness) * (LeftBound65293.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound70611

namespace LeftBound70616
def owner : Owner := ⟨.program ⟨214⟩, ⟨7196⟩⟩
def transferEvent : Nat := 70616
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 70614 .coefficient) (.predecessor 1 70615 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70614 .coefficient)
      LeftBound65164.bound (LeftBound65164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65164.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70615 .coefficient)
      LeftBound11982.bound (LeftBound11982.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11983RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11982.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11982.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound65164.bound LeftBound11982.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65164.bound, LeftBound11982.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound65164.actual selector witness) * (LeftBound11982.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70616

namespace LeftBound70621
def owner : Owner := ⟨.program ⟨214⟩, ⟨11383⟩⟩
def transferEvent : Nat := 70621
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 70619 .coefficient, .predecessor 1 70620 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70619 .coefficient)
      LeftBound70616.bound (LeftBound70616.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events275.exact70618RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70616.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70616.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70620 .coefficient)
      LeftBound70611.bound (LeftBound70611.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events275.exact70613RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70611.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70611.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound70616.bound, LeftBound70611.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70616.bound, LeftBound70611.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound70616.actual selector witness, LeftBound70611.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound70621

namespace LeftBound70625
def owner : Owner := ⟨.program ⟨214⟩, ⟨11384⟩⟩
def transferEvent : Nat := 70625
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 70623 .coefficient, .predecessor 1 70624 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70623 .coefficient)
      LeftBound70621.bound (LeftBound70621.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events275.exact70622RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70621.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70621.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70624 .coefficient)
      LeftBound11974.bound (LeftBound11974.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11975RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11974.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11974.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound70621.bound, LeftBound11974.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70621.bound, LeftBound11974.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound70621.actual selector witness, LeftBound11974.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound70625

namespace LeftBound70626
def owner : Owner := ⟨.program ⟨214⟩, ⟨11384⟩⟩
def transferEvent : Nat := 70626
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨92⟩⟩]⟩ [⟨.result 11975 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11975 .coefficient)
      LeftBound11974.bound (LeftBound11974.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨92⟩⟩) (rawTerms := some (Proof.Events046.exact11975RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11974.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11974.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound11974.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11974.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound11974.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound70626

namespace LeftBound70631
def owner : Owner := ⟨.program ⟨214⟩, ⟨13984⟩⟩
def transferEvent : Nat := 70631
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 70629 .coefficient) (.predecessor 1 70630 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70629 .coefficient)
      LeftBound70625.bound (LeftBound70625.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events275.exact70628RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70625.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70625.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70630 .coefficient)
      LeftAuthority3341.bound (LeftAuthority3341.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events013.exact3342RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3341.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3341.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound70625.bound LeftAuthority3341.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70625.bound, LeftAuthority3341.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound70625.actual selector witness) * (LeftAuthority3341.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70631

namespace LeftBound70632
def owner : Owner := ⟨.program ⟨214⟩, ⟨13984⟩⟩
def transferEvent : Nat := 70632
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨13981⟩⟩], []⟩ [⟨.result 3342 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 3342 .coefficient)
      LeftAuthority3341.bound (LeftAuthority3341.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨13981⟩⟩) (rawTerms := some (Proof.Events013.exact3342RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3341.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3341.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority3341.bound []
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3341.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority3341.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound70632

namespace LeftBound70633
def owner : Owner := ⟨.program ⟨214⟩, ⟨13984⟩⟩
def transferEvent : Nat := 70633
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 70628 .summary) (.transfer 70632) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 70628 .summary)
      LeftBound70626.bound (LeftBound70626.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11384⟩⟩) (rawTerms := some (Proof.Events275.exact70628RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound70626.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 70632)
      LeftBound70632.bound (LeftBound70632.actual selector witness) := by
  exact .transfer (LeftBound70632.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound70626.bound LeftBound70632.bound
def bound : CoeffClass := .finite ⟨13312, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70626.bound, LeftBound70632.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound70626.actual selector witness) * (LeftBound70632.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70633

namespace LeftBound70639
def owner : Owner := ⟨.program ⟨214⟩, ⟨13985⟩⟩
def transferEvent : Nat := 70639
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 70637 .coefficient) (.predecessor 1 70638 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70637 .coefficient)
      LeftAuthority3341.bound (LeftAuthority3341.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events013.exact3342RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3341.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3341.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70638 .coefficient)
      LeftBound65293.bound (LeftBound65293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65293.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority3341.bound LeftBound65293.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3341.bound, LeftBound65293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority3341.actual selector witness) * (LeftBound65293.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound70639

namespace LeftBound70644
def owner : Owner := ⟨.program ⟨214⟩, ⟨7176⟩⟩
def transferEvent : Nat := 70644
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 70642 .coefficient) (.predecessor 1 70643 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70642 .coefficient)
      LeftBound65164.bound (LeftBound65164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65164.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70643 .coefficient)
      LeftBound12023.bound (LeftBound12023.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact12024RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12023.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12023.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound65164.bound LeftBound12023.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65164.bound, LeftBound12023.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound65164.actual selector witness) * (LeftBound12023.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70644

namespace LeftBound70649
def owner : Owner := ⟨.program ⟨214⟩, ⟨13986⟩⟩
def transferEvent : Nat := 70649
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 70647 .coefficient, .predecessor 1 70648 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70647 .coefficient)
      LeftBound70644.bound (LeftBound70644.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events275.exact70646RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70644.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70644.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70648 .coefficient)
      LeftBound70639.bound (LeftBound70639.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events275.exact70641RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70639.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70639.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound70644.bound, LeftBound70639.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70644.bound, LeftBound70639.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound70644.actual selector witness, LeftBound70639.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound70649

namespace LeftBound70653
def owner : Owner := ⟨.program ⟨214⟩, ⟨13987⟩⟩
def transferEvent : Nat := 70653
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 70651 .coefficient, .predecessor 1 70652 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70651 .coefficient)
      LeftBound70649.bound (LeftBound70649.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events275.exact70650RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70649.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70649.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70652 .coefficient)
      LeftBound12015.bound (LeftBound12015.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact12016RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12015.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12015.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound70649.bound, LeftBound12015.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70649.bound, LeftBound12015.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound70649.actual selector witness, LeftBound12015.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound70653

namespace LeftBound70654
def owner : Owner := ⟨.program ⟨214⟩, ⟨13987⟩⟩
def transferEvent : Nat := 70654
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨72⟩⟩]⟩ [⟨.result 12016 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12016 .coefficient)
      LeftBound12015.bound (LeftBound12015.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨72⟩⟩) (rawTerms := some (Proof.Events046.exact12016RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12015.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12015.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound12015.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12015.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound12015.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound70654

namespace LeftBound70659
def owner : Owner := ⟨.program ⟨214⟩, ⟨13988⟩⟩
def transferEvent : Nat := 70659
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 70657 .coefficient) (.predecessor 1 70658 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70657 .coefficient)
      LeftBound70653.bound (LeftBound70653.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events276.exact70656RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70653.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70653.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70658 .coefficient)
      LeftBound12012.bound (LeftBound12012.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact12013RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12012.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12012.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound70653.bound LeftBound12012.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70653.bound, LeftBound12012.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound70653.actual selector witness) * (LeftBound12012.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70659

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
