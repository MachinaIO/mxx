import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard052
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard538
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard541
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard562

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound82747
def owner : Owner := ⟨.program ⟨214⟩, ⟨16508⟩⟩
def transferEvent : Nat := 82747
def frameStart : Nat := 82674
def rule : BoundRule := .sum [.predecessor 0 82745 .coefficient, .predecessor 1 82746 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82745 .coefficient)
      LeftAuthority82743.bound (LeftAuthority82743.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events323.exact82744RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82743.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82743.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82746 .coefficient)
      LeftBound82739.bound (LeftBound82739.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events323.exact82741RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82739.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82739.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority82743.bound, LeftBound82739.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority82743.bound, LeftBound82739.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority82743.actual selector witness, LeftBound82739.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound82747

namespace LeftBound82751
def owner : Owner := ⟨.program ⟨214⟩, ⟨28952⟩⟩
def transferEvent : Nat := 82751
def frameStart : Nat := 82674
def rule : BoundRule := .product (.predecessor 0 82749 .coefficient) (.predecessor 1 82750 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82749 .coefficient)
      LeftBound82747.bound (LeftBound82747.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events323.exact82748RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82747.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82747.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82750 .coefficient)
      LeftAuthority82724.bound (LeftAuthority82724.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events323.exact82725RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82724.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82724.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound82747.bound LeftAuthority82724.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82747.bound, LeftAuthority82724.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound82747.actual selector witness) * (LeftAuthority82724.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82751

namespace LeftBound82762
def owner : Owner := ⟨.program ⟨214⟩, ⟨17905⟩⟩
def transferEvent : Nat := 82762
def frameStart : Nat := 82674
def rule : BoundRule := .product (.predecessor 0 82760 .coefficient) (.predecessor 1 82761 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82760 .coefficient)
      LeftAuthority82735.bound (LeftAuthority82735.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events323.exact82736RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82735.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82735.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82761 .coefficient)
      LeftAuthority82758.bound (LeftAuthority82758.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events323.exact82759RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82758.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82758.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority82735.bound LeftAuthority82758.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority82735.bound, LeftAuthority82758.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority82735.actual selector witness) * (LeftAuthority82758.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82762

namespace LeftBound82770
def owner : Owner := ⟨.program ⟨214⟩, ⟨17906⟩⟩
def transferEvent : Nat := 82770
def frameStart : Nat := 82674
def rule : BoundRule := .sum [.predecessor 0 82768 .coefficient, .predecessor 1 82769 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82768 .coefficient)
      LeftAuthority82766.bound (LeftAuthority82766.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events323.exact82767RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82766.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82766.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82769 .coefficient)
      LeftBound82762.bound (LeftBound82762.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events323.exact82764RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82762.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82762.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority82766.bound, LeftBound82762.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority82766.bound, LeftBound82762.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority82766.actual selector witness, LeftBound82762.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound82770

namespace LeftBound82774
def owner : Owner := ⟨.program ⟨214⟩, ⟨28956⟩⟩
def transferEvent : Nat := 82774
def frameStart : Nat := 82674
def rule : BoundRule := .sum [.predecessor 0 82772 .coefficient, .predecessor 1 82773 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82772 .coefficient)
      LeftBound82770.bound (LeftBound82770.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events323.exact82771RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82770.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82770.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82773 .coefficient)
      LeftBound82751.bound (LeftBound82751.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events323.exact82756RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82751.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82751.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound82770.bound, LeftBound82751.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82770.bound, LeftBound82751.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound82770.actual selector witness, LeftBound82751.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound82774

namespace LeftBound82787
def owner : Owner := ⟨.program ⟨214⟩, ⟨28954⟩⟩
def transferEvent : Nat := 82787
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 82785 .coefficient, .predecessor 1 82786 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82785 .coefficient)
      LeftBound82616.bound (LeftBound82616.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events323.exact82784RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82616.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82616.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82786 .coefficient)
      LeftBound82599.bound (LeftBound82599.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events322.exact82606RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82599.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82599.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound82616.bound, LeftBound82599.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82616.bound, LeftBound82599.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound82616.actual selector witness, LeftBound82599.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound82787

namespace LeftBound82790
def owner : Owner := ⟨.program ⟨214⟩, ⟨28954⟩⟩
def transferEvent : Nat := 82790
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 82784 .summary, .result 82606 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 82784 .summary)
      LeftBound82618.bound (LeftBound82618.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22123⟩⟩) (rawTerms := some (Proof.Events323.exact82784RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound82618.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 82606 .summary)
      LeftBound82601.bound (LeftBound82601.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28953⟩⟩) (rawTerms := some (Proof.Events322.exact82606RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound82601.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound82618.bound, LeftBound82601.bound]
def bound : CoeffClass := .finite ⟨1292315010834812776448, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82618.bound, LeftBound82601.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound82618.actual selector witness, LeftBound82601.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound82790

namespace LeftBound82814
def owner : Owner := ⟨.program ⟨214⟩, ⟨11960⟩⟩
def transferEvent : Nat := 82814
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 82812 .coefficient) (.predecessor 1 82813 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82812 .coefficient)
      LeftAuthority3965.bound (LeftAuthority3965.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events015.exact3966RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3965.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3965.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82813 .coefficient)
      LeftBound79918.bound (LeftBound79918.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79920RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79918.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79918.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority3965.bound LeftBound79918.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3965.bound, LeftBound79918.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority3965.actual selector witness) * (LeftBound79918.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound82814

namespace LeftBound82819
def owner : Owner := ⟨.program ⟨214⟩, ⟨7240⟩⟩
def transferEvent : Nat := 82819
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 82817 .coefficient) (.predecessor 1 82818 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82817 .coefficient)
      LeftBound79789.bound (LeftBound79789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events311.exact79790RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79789.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79789.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82818 .coefficient)
      LeftBound9477.bound (LeftBound9477.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9478RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9477.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9477.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound79789.bound LeftBound9477.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79789.bound, LeftBound9477.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound79789.actual selector witness) * (LeftBound9477.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82819

namespace LeftBound82824
def owner : Owner := ⟨.program ⟨214⟩, ⟨11961⟩⟩
def transferEvent : Nat := 82824
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 82822 .coefficient, .predecessor 1 82823 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82822 .coefficient)
      LeftBound82819.bound (LeftBound82819.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events323.exact82821RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82819.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82819.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82823 .coefficient)
      LeftBound82814.bound (LeftBound82814.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events323.exact82816RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82814.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82814.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound82819.bound, LeftBound82814.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82819.bound, LeftBound82814.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound82819.actual selector witness, LeftBound82814.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound82824

namespace LeftBound82828
def owner : Owner := ⟨.program ⟨214⟩, ⟨11962⟩⟩
def transferEvent : Nat := 82828
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 82826 .coefficient, .predecessor 1 82827 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82826 .coefficient)
      LeftBound82824.bound (LeftBound82824.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events323.exact82825RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82824.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82824.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82827 .coefficient)
      LeftBound9469.bound (LeftBound9469.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events036.exact9470RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9469.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9469.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound82824.bound, LeftBound9469.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82824.bound, LeftBound9469.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound82824.actual selector witness, LeftBound9469.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound82828

namespace LeftBound82829
def owner : Owner := ⟨.program ⟨214⟩, ⟨11962⟩⟩
def transferEvent : Nat := 82829
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨98⟩⟩]⟩ [⟨.result 9470 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9470 .coefficient)
      LeftBound9469.bound (LeftBound9469.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨98⟩⟩) (rawTerms := some (Proof.Events036.exact9470RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9469.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9469.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound9469.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9469.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound9469.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound82829

namespace LeftBound82834
def owner : Owner := ⟨.program ⟨214⟩, ⟨11963⟩⟩
def transferEvent : Nat := 82834
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 82832 .coefficient) (.predecessor 1 82833 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82832 .coefficient)
      LeftBound82828.bound (LeftBound82828.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events323.exact82831RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82828.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82828.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82833 .coefficient)
      LeftAuthority3968.bound (LeftAuthority3968.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events015.exact3969RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3968.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3968.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound82828.bound LeftAuthority3968.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82828.bound, LeftAuthority3968.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound82828.actual selector witness) * (LeftAuthority3968.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82834

namespace LeftBound82835
def owner : Owner := ⟨.program ⟨214⟩, ⟨11963⟩⟩
def transferEvent : Nat := 82835
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨9715⟩⟩], []⟩ [⟨.result 3969 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 3969 .coefficient)
      LeftAuthority3968.bound (LeftAuthority3968.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨9715⟩⟩) (rawTerms := some (Proof.Events015.exact3969RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3968.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3968.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority3968.bound []
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3968.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority3968.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound82835

namespace LeftBound82836
def owner : Owner := ⟨.program ⟨214⟩, ⟨11963⟩⟩
def transferEvent : Nat := 82836
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 82831 .summary) (.transfer 82835) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 82831 .summary)
      LeftBound82829.bound (LeftBound82829.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11962⟩⟩) (rawTerms := some (Proof.Events323.exact82831RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound82829.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 82835)
      LeftBound82835.bound (LeftBound82835.actual selector witness) := by
  exact .transfer (LeftBound82835.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound82829.bound LeftBound82835.bound
def bound : CoeffClass := .finite ⟨29952, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82829.bound, LeftBound82835.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound82829.actual selector witness) * (LeftBound82835.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82836

namespace LeftBound82842
def owner : Owner := ⟨.program ⟨214⟩, ⟨9716⟩⟩
def transferEvent : Nat := 82842
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 82840 .coefficient) (.predecessor 1 82841 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82840 .coefficient)
      LeftAuthority3968.bound (LeftAuthority3968.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events015.exact3969RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3968.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3968.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82841 .coefficient)
      LeftBound79918.bound (LeftBound79918.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79920RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79918.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79918.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority3968.bound LeftBound79918.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3968.bound, LeftBound79918.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority3968.actual selector witness) * (LeftBound79918.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound82842

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
