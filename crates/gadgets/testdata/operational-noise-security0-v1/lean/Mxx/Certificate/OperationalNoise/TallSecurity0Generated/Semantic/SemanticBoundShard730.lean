import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard696
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard697
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard729

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound106537
def owner : Owner := ⟨.program ⟨214⟩, ⟨15456⟩⟩
def transferEvent : Nat := 106537
def frameStart : Nat := 106484
def rule : BoundRule := .product (.predecessor 0 106535 .coefficient) (.predecessor 1 106536 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106535 .coefficient)
      LeftAuthority106533.bound (LeftAuthority106533.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events416.exact106534RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106533.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106533.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106536 .coefficient)
      LeftBound106531.bound (LeftBound106531.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events416.exact106532RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106531.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106531.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority106533.bound LeftBound106531.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106533.bound, LeftBound106531.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority106533.actual selector witness) * (LeftBound106531.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106537

namespace LeftBound106545
def owner : Owner := ⟨.program ⟨214⟩, ⟨15457⟩⟩
def transferEvent : Nat := 106545
def frameStart : Nat := 106484
def rule : BoundRule := .sum [.predecessor 0 106543 .coefficient, .predecessor 1 106544 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106543 .coefficient)
      LeftAuthority106541.bound (LeftAuthority106541.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events416.exact106542RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106541.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106541.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106544 .coefficient)
      LeftBound106537.bound (LeftBound106537.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events416.exact106539RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106537.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106537.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority106541.bound, LeftBound106537.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106541.bound, LeftBound106537.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority106541.actual selector witness, LeftBound106537.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound106545

namespace LeftBound106549
def owner : Owner := ⟨.program ⟨214⟩, ⟨26957⟩⟩
def transferEvent : Nat := 106549
def frameStart : Nat := 106484
def rule : BoundRule := .product (.predecessor 0 106547 .coefficient) (.predecessor 1 106548 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106547 .coefficient)
      LeftBound106545.bound (LeftBound106545.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events416.exact106546RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106545.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106545.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106548 .coefficient)
      LeftAuthority106522.bound (LeftAuthority106522.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events416.exact106523RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106522.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106522.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound106545.bound LeftAuthority106522.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106545.bound, LeftAuthority106522.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound106545.actual selector witness) * (LeftAuthority106522.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106549

namespace LeftBound106560
def owner : Owner := ⟨.program ⟨214⟩, ⟨15506⟩⟩
def transferEvent : Nat := 106560
def frameStart : Nat := 106484
def rule : BoundRule := .product (.predecessor 0 106558 .coefficient) (.predecessor 1 106559 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106558 .coefficient)
      LeftAuthority106533.bound (LeftAuthority106533.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events416.exact106534RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106533.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106533.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106559 .coefficient)
      LeftAuthority106556.bound (LeftAuthority106556.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events416.exact106557RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106556.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106556.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority106533.bound LeftAuthority106556.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106533.bound, LeftAuthority106556.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority106533.actual selector witness) * (LeftAuthority106556.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106560

namespace LeftBound106568
def owner : Owner := ⟨.program ⟨214⟩, ⟨15507⟩⟩
def transferEvent : Nat := 106568
def frameStart : Nat := 106484
def rule : BoundRule := .sum [.predecessor 0 106566 .coefficient, .predecessor 1 106567 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106566 .coefficient)
      LeftAuthority106564.bound (LeftAuthority106564.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events416.exact106565RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106564.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106564.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106567 .coefficient)
      LeftBound106560.bound (LeftBound106560.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events416.exact106562RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106560.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106560.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority106564.bound, LeftBound106560.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106564.bound, LeftBound106560.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority106564.actual selector witness, LeftBound106560.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound106568

namespace LeftBound106572
def owner : Owner := ⟨.program ⟨214⟩, ⟨26962⟩⟩
def transferEvent : Nat := 106572
def frameStart : Nat := 106484
def rule : BoundRule := .sum [.predecessor 0 106570 .coefficient, .predecessor 1 106571 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106570 .coefficient)
      LeftBound106568.bound (LeftBound106568.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events416.exact106569RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106568.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106568.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106571 .coefficient)
      LeftBound106549.bound (LeftBound106549.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events416.exact106554RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106549.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106549.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound106568.bound, LeftBound106549.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106568.bound, LeftBound106549.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound106568.actual selector witness, LeftBound106549.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound106572

namespace LeftBound106585
def owner : Owner := ⟨.program ⟨214⟩, ⟨26959⟩⟩
def transferEvent : Nat := 106585
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 106583 .coefficient, .predecessor 1 106584 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106583 .coefficient)
      LeftBound106438.bound (LeftBound106438.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events416.exact106582RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106438.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106438.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106584 .coefficient)
      LeftBound106421.bound (LeftBound106421.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106428RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106421.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106421.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound106438.bound, LeftBound106421.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106438.bound, LeftBound106421.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound106438.actual selector witness, LeftBound106421.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound106585

namespace LeftBound106588
def owner : Owner := ⟨.program ⟨214⟩, ⟨26959⟩⟩
def transferEvent : Nat := 106588
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 106582 .summary, .result 106428 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 106582 .summary)
      LeftBound106440.bound (LeftBound106440.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20744⟩⟩) (rawTerms := some (Proof.Events416.exact106582RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound106440.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 106428 .summary)
      LeftBound106423.bound (LeftBound106423.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26958⟩⟩) (rawTerms := some (Proof.Events415.exact106428RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound106423.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound106440.bound, LeftBound106423.bound]
def bound : CoeffClass := .finite ⟨1291933999269462814720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106440.bound, LeftBound106423.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound106440.actual selector witness, LeftBound106423.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound106588

namespace LeftBound106592
def owner : Owner := ⟨.program ⟨214⟩, ⟨26960⟩⟩
def transferEvent : Nat := 106592
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 106590 .coefficient) (.predecessor 1 106591 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106590 .coefficient)
      LeftBound106585.bound (LeftBound106585.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events416.exact106589RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106585.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106585.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106591 .coefficient)
      LeftBound5798.bound (LeftBound5798.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5799RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5798.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5798.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound106585.bound LeftBound5798.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106585.bound, LeftBound5798.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound106585.actual selector witness) * (LeftBound5798.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106592

namespace LeftBound106593
def owner : Owner := ⟨.program ⟨214⟩, ⟨26960⟩⟩
def transferEvent : Nat := 106593
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩ [⟨.result 5795 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5795 .coefficient)
      LeftAuthority5794.bound (LeftAuthority5794.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6655⟩⟩) (rawTerms := some (Proof.Events022.exact5795RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5794.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5794.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5794.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5794.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5794.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound106593

namespace LeftBound106594
def owner : Owner := ⟨.program ⟨214⟩, ⟨26960⟩⟩
def transferEvent : Nat := 106594
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 106589 .summary) (.transfer 106593) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 106589 .summary)
      LeftBound106588.bound (LeftBound106588.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26959⟩⟩) (rawTerms := some (Proof.Events416.exact106589RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound106588.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 106593)
      LeftBound106593.bound (LeftBound106593.actual selector witness) := by
  exact .transfer (LeftBound106593.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound106588.bound LeftBound106593.bound
def bound : CoeffClass := .finite ⟨4741418448262916841427435520, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106588.bound, LeftBound106593.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound106588.actual selector witness) * (LeftBound106593.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106594

namespace LeftBound106609
def owner : Owner := ⟨.program ⟨214⟩, ⟨26741⟩⟩
def transferEvent : Nat := 106609
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 106607 .coefficient) (.predecessor 1 106608 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106607 .coefficient)
      LeftBound101130.bound (LeftBound101130.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events395.exact101134RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101130.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101130.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106608 .coefficient)
      LeftAuthority106605.bound (LeftAuthority106605.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events416.exact106606RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106605.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106605.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound101130.bound LeftAuthority106605.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101130.bound, LeftAuthority106605.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound101130.actual selector witness) * (LeftAuthority106605.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106609

namespace LeftBound106610
def owner : Owner := ⟨.program ⟨214⟩, ⟨26741⟩⟩
def transferEvent : Nat := 106610
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26739⟩⟩]⟩ [⟨.result 106606 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 106606 .coefficient)
      LeftAuthority106605.bound (LeftAuthority106605.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26739⟩⟩) (rawTerms := some (Proof.Events416.exact106606RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106605.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106605.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority106605.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106605.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority106605.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound106610

namespace LeftBound106611
def owner : Owner := ⟨.program ⟨214⟩, ⟨26741⟩⟩
def transferEvent : Nat := 106611
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 101134 .summary) (.transfer 106610) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 101134 .summary)
      LeftBound101133.bound (LeftBound101133.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25054⟩⟩) (rawTerms := some (Proof.Events395.exact101134RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound101133.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 106610)
      LeftBound106610.bound (LeftBound106610.actual selector witness) := by
  exact .transfer (LeftBound106610.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound101133.bound LeftBound106610.bound
def bound : CoeffClass := .finite ⟨1291911585013138718720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101133.bound, LeftBound106610.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound101133.actual selector witness) * (LeftBound106610.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106611

namespace LeftBound106622
def owner : Owner := ⟨.program ⟨214⟩, ⟨20599⟩⟩
def transferEvent : Nat := 106622
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 106620 .coefficient) (.value (.predecessor 1 106621 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106620 .coefficient)
      LeftAuthority106618.bound (LeftAuthority106618.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events416.exact106619RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106618.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106618.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106621 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority106618.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106618.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority106618.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound106622

namespace LeftBound106626
def owner : Owner := ⟨.program ⟨214⟩, ⟨20600⟩⟩
def transferEvent : Nat := 106626
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 106624 .coefficient) (.predecessor 1 106625 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106624 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106625 .coefficient)
      LeftBound106622.bound (LeftBound106622.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events416.exact106623RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106622.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106622.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound106622.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound106622.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound106622.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106626

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
