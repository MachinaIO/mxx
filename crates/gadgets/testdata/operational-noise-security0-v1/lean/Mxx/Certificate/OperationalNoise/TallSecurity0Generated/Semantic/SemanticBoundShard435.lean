import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard409
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard410
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard412
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard413
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard414
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard416
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard417
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard418
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard420
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard434

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound64864
def owner : Owner := ⟨.program ⟨214⟩, ⟨28528⟩⟩
def transferEvent : Nat := 64864
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 64860 .summary, .result 62663 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 64860 .summary)
      LeftBound64859.bound (LeftBound64859.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28311⟩⟩) (rawTerms := some (Proof.Events253.exact64860RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound64859.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 62663 .summary)
      LeftBound62658.bound (LeftBound62658.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28527⟩⟩) (rawTerms := some (Proof.Events244.exact62663RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound62658.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64859.bound, LeftBound62658.bound]
def bound : CoeffClass := .finite ⟨52159098727244633768554659892, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64859.bound, LeftBound62658.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64859.actual selector witness, LeftBound62658.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64864

namespace LeftBound64868
def owner : Owner := ⟨.program ⟨214⟩, ⟨28745⟩⟩
def transferEvent : Nat := 64868
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 64866 .coefficient, .predecessor 1 64867 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64866 .coefficient)
      LeftBound64863.bound (LeftBound64863.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64865RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64863.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64863.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64867 .coefficient)
      LeftBound62444.bound (LeftBound62444.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events243.exact62451RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62444.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62444.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64863.bound, LeftBound62444.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64863.bound, LeftBound62444.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64863.actual selector witness, LeftBound62444.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64868

namespace LeftBound64869
def owner : Owner := ⟨.program ⟨214⟩, ⟨28745⟩⟩
def transferEvent : Nat := 64869
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 64865 .summary, .result 62451 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 64865 .summary)
      LeftBound64864.bound (LeftBound64864.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28528⟩⟩) (rawTerms := some (Proof.Events253.exact64865RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound64864.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 62451 .summary)
      LeftBound62446.bound (LeftBound62446.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28744⟩⟩) (rawTerms := some (Proof.Events243.exact62451RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound62446.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64864.bound, LeftBound62446.bound]
def bound : CoeffClass := .finite ⟨56901750985984920673341931572, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64864.bound, LeftBound62446.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64864.actual selector witness, LeftBound62446.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64869

namespace LeftBound64873
def owner : Owner := ⟨.program ⟨214⟩, ⟨28962⟩⟩
def transferEvent : Nat := 64873
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 64871 .coefficient, .predecessor 1 64872 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64871 .coefficient)
      LeftBound64868.bound (LeftBound64868.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64870RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64868.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64868.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64872 .coefficient)
      LeftBound62232.bound (LeftBound62232.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events243.exact62239RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62232.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62232.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64868.bound, LeftBound62232.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64868.bound, LeftBound62232.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64868.actual selector witness, LeftBound62232.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64873

namespace LeftBound64874
def owner : Owner := ⟨.program ⟨214⟩, ⟨28962⟩⟩
def transferEvent : Nat := 64874
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 64870 .summary, .result 62239 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 64870 .summary)
      LeftBound64869.bound (LeftBound64869.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28745⟩⟩) (rawTerms := some (Proof.Events253.exact64870RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound64869.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 62239 .summary)
      LeftBound62234.bound (LeftBound62234.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28961⟩⟩) (rawTerms := some (Proof.Events243.exact62239RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound62234.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64869.bound, LeftBound62234.bound]
def bound : CoeffClass := .finite ⟨61644567752788856919910514740, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64869.bound, LeftBound62234.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64869.actual selector witness, LeftBound62234.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64874

namespace LeftBound64878
def owner : Owner := ⟨.program ⟨214⟩, ⟨29179⟩⟩
def transferEvent : Nat := 64878
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 64876 .coefficient, .predecessor 1 64877 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64876 .coefficient)
      LeftBound64873.bound (LeftBound64873.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64875RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64873.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64873.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64877 .coefficient)
      LeftBound62020.bound (LeftBound62020.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events242.exact62027RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62020.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62020.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64873.bound, LeftBound62020.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64873.bound, LeftBound62020.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64873.actual selector witness, LeftBound62020.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64878

namespace LeftBound64879
def owner : Owner := ⟨.program ⟨214⟩, ⟨29179⟩⟩
def transferEvent : Nat := 64879
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 64875 .summary, .result 62027 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 64875 .summary)
      LeftBound64874.bound (LeftBound64874.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28962⟩⟩) (rawTerms := some (Proof.Events253.exact64875RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound64874.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 62027 .summary)
      LeftBound62022.bound (LeftBound62022.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29178⟩⟩) (rawTerms := some (Proof.Events242.exact62027RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound62022.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64874.bound, LeftBound62022.bound]
def bound : CoeffClass := .finite ⟨66387466773624617837369753652, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64874.bound, LeftBound62022.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64874.actual selector witness, LeftBound62022.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64879

namespace LeftBound64883
def owner : Owner := ⟨.program ⟨214⟩, ⟨29396⟩⟩
def transferEvent : Nat := 64883
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 64881 .coefficient, .predecessor 1 64882 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64881 .coefficient)
      LeftBound64878.bound (LeftBound64878.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64880RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64878.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64878.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64882 .coefficient)
      LeftBound61808.bound (LeftBound61808.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events241.exact61815RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61808.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61808.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64878.bound, LeftBound61808.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64878.bound, LeftBound61808.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64878.actual selector witness, LeftBound61808.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64883

namespace LeftBound64884
def owner : Owner := ⟨.program ⟨214⟩, ⟨29396⟩⟩
def transferEvent : Nat := 64884
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 64880 .summary, .result 61815 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 64880 .summary)
      LeftBound64879.bound (LeftBound64879.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29179⟩⟩) (rawTerms := some (Proof.Events253.exact64880RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound64879.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 61815 .summary)
      LeftBound61810.bound (LeftBound61810.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29395⟩⟩) (rawTerms := some (Proof.Events241.exact61815RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound61810.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64879.bound, LeftBound61810.bound]
def bound : CoeffClass := .finite ⟨71130530302524028096610304052, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64879.bound, LeftBound61810.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64879.actual selector witness, LeftBound61810.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64884

namespace LeftBound64888
def owner : Owner := ⟨.program ⟨214⟩, ⟨29613⟩⟩
def transferEvent : Nat := 64888
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 64886 .coefficient, .predecessor 1 64887 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64886 .coefficient)
      LeftBound64883.bound (LeftBound64883.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64885RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64883.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64883.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64887 .coefficient)
      LeftBound61596.bound (LeftBound61596.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events240.exact61603RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61596.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61596.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64883.bound, LeftBound61596.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64883.bound, LeftBound61596.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64883.actual selector witness, LeftBound61596.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64888

namespace LeftBound64889
def owner : Owner := ⟨.program ⟨214⟩, ⟨29613⟩⟩
def transferEvent : Nat := 64889
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 64885 .summary, .result 61603 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 64885 .summary)
      LeftBound64884.bound (LeftBound64884.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29396⟩⟩) (rawTerms := some (Proof.Events253.exact64885RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound64884.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 61603 .summary)
      LeftBound61598.bound (LeftBound61598.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29612⟩⟩) (rawTerms := some (Proof.Events240.exact61603RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound61598.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64884.bound, LeftBound61598.bound]
def bound : CoeffClass := .finite ⟨75873840593518912368522821684, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64884.bound, LeftBound61598.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64884.actual selector witness, LeftBound61598.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64889

namespace LeftBound64893
def owner : Owner := ⟨.program ⟨214⟩, ⟨29830⟩⟩
def transferEvent : Nat := 64893
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 64891 .coefficient, .predecessor 1 64892 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64891 .coefficient)
      LeftBound64888.bound (LeftBound64888.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64890RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64888.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64888.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64892 .coefficient)
      LeftBound61384.bound (LeftBound61384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events239.exact61391RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61384.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64888.bound, LeftBound61384.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64888.bound, LeftBound61384.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64888.actual selector witness, LeftBound61384.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64893

namespace LeftBound64894
def owner : Owner := ⟨.program ⟨214⟩, ⟨29830⟩⟩
def transferEvent : Nat := 64894
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 64890 .summary, .result 61391 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 64890 .summary)
      LeftBound64889.bound (LeftBound64889.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29613⟩⟩) (rawTerms := some (Proof.Events253.exact64890RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound64889.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 61391 .summary)
      LeftBound61386.bound (LeftBound61386.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29829⟩⟩) (rawTerms := some (Proof.Events239.exact61391RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound61386.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64889.bound, LeftBound61386.bound]
def bound : CoeffClass := .finite ⟨80617397646609270653107306548, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64889.bound, LeftBound61386.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64889.actual selector witness, LeftBound61386.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64894

namespace LeftBound64898
def owner : Owner := ⟨.program ⟨214⟩, ⟨30137⟩⟩
def transferEvent : Nat := 64898
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 64896 .coefficient, .predecessor 1 64897 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64896 .coefficient)
      LeftBound64893.bound (LeftBound64893.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64895RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64893.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64893.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64897 .coefficient)
      LeftBound61172.bound (LeftBound61172.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events238.exact61179RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61172.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61172.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64893.bound, LeftBound61172.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64893.bound, LeftBound61172.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64893.actual selector witness, LeftBound61172.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64898

namespace LeftBound64899
def owner : Owner := ⟨.program ⟨214⟩, ⟨30137⟩⟩
def transferEvent : Nat := 64899
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 64895 .summary, .result 61179 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 64895 .summary)
      LeftBound64894.bound (LeftBound64894.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29830⟩⟩) (rawTerms := some (Proof.Events253.exact64895RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound64894.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 61179 .summary)
      LeftBound61174.bound (LeftBound61174.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30136⟩⟩) (rawTerms := some (Proof.Events238.exact61179RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound61174.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64894.bound, LeftBound61174.bound]
def bound : CoeffClass := .finite ⟨85361036953731453608582447156, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64894.bound, LeftBound61174.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64894.actual selector witness, LeftBound61174.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64899

namespace LeftBound64903
def owner : Owner := ⟨.program ⟨214⟩, ⟨30148⟩⟩
def transferEvent : Nat := 64903
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 64901 .coefficient, .predecessor 1 64902 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64901 .coefficient)
      LeftBound64898.bound (LeftBound64898.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64900RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64898.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64898.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64902 .coefficient)
      LeftBound60960.bound (LeftBound60960.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events238.exact60967RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60960.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60960.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64898.bound, LeftBound60960.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64898.bound, LeftBound60960.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64898.actual selector witness, LeftBound60960.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64903

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
