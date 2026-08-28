import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard545
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard548
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard552
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard556
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard559
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard563
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard566
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard567
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard570
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard574
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard607

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound88595
def owner : Owner := ⟨.program ⟨214⟩, ⟨28304⟩⟩
def transferEvent : Nat := 88595
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 88591 .summary, .result 84231 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 88591 .summary)
      LeftBound88590.bound (LeftBound88590.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28087⟩⟩) (rawTerms := some (Proof.Events346.exact88591RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound88590.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 84231 .summary)
      LeftBound84230.bound (LeftBound84230.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28303⟩⟩) (rawTerms := some (Proof.Events329.exact84231RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84230.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88590.bound, LeftBound84230.bound]
def bound : CoeffClass := .finite ⟨12920023572267756019712, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88590.bound, LeftBound84230.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88590.actual selector witness, LeftBound84230.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88595

namespace LeftBound88599
def owner : Owner := ⟨.program ⟨214⟩, ⟨28521⟩⟩
def transferEvent : Nat := 88599
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 88597 .coefficient, .predecessor 1 88598 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88597 .coefficient)
      LeftBound88594.bound (LeftBound88594.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events346.exact88596RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88594.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88594.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88598 .coefficient)
      LeftBound83747.bound (LeftBound83747.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83751RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83747.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83747.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88594.bound, LeftBound83747.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88594.bound, LeftBound83747.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88594.actual selector witness, LeftBound83747.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88599

namespace LeftBound88600
def owner : Owner := ⟨.program ⟨214⟩, ⟨28521⟩⟩
def transferEvent : Nat := 88600
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 88596 .summary, .result 83751 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 88596 .summary)
      LeftBound88595.bound (LeftBound88595.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28304⟩⟩) (rawTerms := some (Proof.Events346.exact88596RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound88595.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 83751 .summary)
      LeftBound83750.bound (LeftBound83750.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28520⟩⟩) (rawTerms := some (Proof.Events327.exact83751RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound83750.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88595.bound, LeftBound83750.bound]
def bound : CoeffClass := .finite ⟨14212226520877465866240, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88595.bound, LeftBound83750.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88595.actual selector witness, LeftBound83750.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88600

namespace LeftBound88604
def owner : Owner := ⟨.program ⟨214⟩, ⟨28738⟩⟩
def transferEvent : Nat := 88604
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 88602 .coefficient, .predecessor 1 88603 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88602 .coefficient)
      LeftBound88599.bound (LeftBound88599.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events346.exact88601RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88599.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88599.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88603 .coefficient)
      LeftBound83267.bound (LeftBound83267.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events325.exact83271RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83267.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83267.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88599.bound, LeftBound83267.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88599.bound, LeftBound83267.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88599.actual selector witness, LeftBound83267.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88604

namespace LeftBound88605
def owner : Owner := ⟨.program ⟨214⟩, ⟨28738⟩⟩
def transferEvent : Nat := 88605
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 88601 .summary, .result 83271 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 88601 .summary)
      LeftBound88600.bound (LeftBound88600.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28521⟩⟩) (rawTerms := some (Proof.Events346.exact88601RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound88600.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 83271 .summary)
      LeftBound83270.bound (LeftBound83270.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28737⟩⟩) (rawTerms := some (Proof.Events325.exact83271RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound83270.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88600.bound, LeftBound83270.bound]
def bound : CoeffClass := .finite ⟨15504496706822237470720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88600.bound, LeftBound83270.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88600.actual selector witness, LeftBound83270.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88605

namespace LeftBound88609
def owner : Owner := ⟨.program ⟨214⟩, ⟨28955⟩⟩
def transferEvent : Nat := 88609
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 88607 .coefficient, .predecessor 1 88608 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88607 .coefficient)
      LeftBound88604.bound (LeftBound88604.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events346.exact88606RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88604.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88604.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88608 .coefficient)
      LeftBound82787.bound (LeftBound82787.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events323.exact82791RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82787.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82787.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88604.bound, LeftBound82787.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88604.bound, LeftBound82787.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88604.actual selector witness, LeftBound82787.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88609

namespace LeftBound88610
def owner : Owner := ⟨.program ⟨214⟩, ⟨28955⟩⟩
def transferEvent : Nat := 88610
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 88606 .summary, .result 82791 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 88606 .summary)
      LeftBound88605.bound (LeftBound88605.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28738⟩⟩) (rawTerms := some (Proof.Events346.exact88606RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound88605.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 82791 .summary)
      LeftBound82790.bound (LeftBound82790.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28954⟩⟩) (rawTerms := some (Proof.Events323.exact82791RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound82790.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88605.bound, LeftBound82790.bound]
def bound : CoeffClass := .finite ⟨16796811717657050247168, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88605.bound, LeftBound82790.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88605.actual selector witness, LeftBound82790.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88610

namespace LeftBound88614
def owner : Owner := ⟨.program ⟨214⟩, ⟨29172⟩⟩
def transferEvent : Nat := 88614
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 88612 .coefficient, .predecessor 1 88613 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88612 .coefficient)
      LeftBound88609.bound (LeftBound88609.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events346.exact88611RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88609.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88609.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88613 .coefficient)
      LeftBound82307.bound (LeftBound82307.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82311RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82307.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82307.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88609.bound, LeftBound82307.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88609.bound, LeftBound82307.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88609.actual selector witness, LeftBound82307.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88614

namespace LeftBound88615
def owner : Owner := ⟨.program ⟨214⟩, ⟨29172⟩⟩
def transferEvent : Nat := 88615
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 88611 .summary, .result 82311 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 88611 .summary)
      LeftBound88610.bound (LeftBound88610.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28955⟩⟩) (rawTerms := some (Proof.Events346.exact88611RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound88610.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 82311 .summary)
      LeftBound82310.bound (LeftBound82310.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29171⟩⟩) (rawTerms := some (Proof.Events321.exact82311RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound82310.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88610.bound, LeftBound82310.bound]
def bound : CoeffClass := .finite ⟨18089149140936883609600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88610.bound, LeftBound82310.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88610.actual selector witness, LeftBound82310.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88615

namespace LeftBound88619
def owner : Owner := ⟨.program ⟨214⟩, ⟨29389⟩⟩
def transferEvent : Nat := 88619
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 88617 .coefficient, .predecessor 1 88618 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88617 .coefficient)
      LeftBound88614.bound (LeftBound88614.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events346.exact88616RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88614.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88614.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88618 .coefficient)
      LeftBound81827.bound (LeftBound81827.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events319.exact81831RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81827.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81827.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88614.bound, LeftBound81827.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88614.bound, LeftBound81827.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88614.actual selector witness, LeftBound81827.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88619

namespace LeftBound88620
def owner : Owner := ⟨.program ⟨214⟩, ⟨29389⟩⟩
def transferEvent : Nat := 88620
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 88616 .summary, .result 81831 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 88616 .summary)
      LeftBound88615.bound (LeftBound88615.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29172⟩⟩) (rawTerms := some (Proof.Events346.exact88616RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound88615.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 81831 .summary)
      LeftBound81830.bound (LeftBound81830.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29388⟩⟩) (rawTerms := some (Proof.Events319.exact81831RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound81830.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88615.bound, LeftBound81830.bound]
def bound : CoeffClass := .finite ⟨19381531389106758144000, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88615.bound, LeftBound81830.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88615.actual selector witness, LeftBound81830.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88620

namespace LeftBound88624
def owner : Owner := ⟨.program ⟨214⟩, ⟨29606⟩⟩
def transferEvent : Nat := 88624
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 88622 .coefficient, .predecessor 1 88623 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88622 .coefficient)
      LeftBound88619.bound (LeftBound88619.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events346.exact88621RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88619.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88619.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88623 .coefficient)
      LeftBound81347.bound (LeftBound81347.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events317.exact81351RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81347.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81347.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88619.bound, LeftBound81347.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88619.bound, LeftBound81347.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88619.actual selector witness, LeftBound81347.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88624

namespace LeftBound88625
def owner : Owner := ⟨.program ⟨214⟩, ⟨29606⟩⟩
def transferEvent : Nat := 88625
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 88621 .summary, .result 81351 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 88621 .summary)
      LeftBound88620.bound (LeftBound88620.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29389⟩⟩) (rawTerms := some (Proof.Events346.exact88621RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound88620.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 81351 .summary)
      LeftBound81350.bound (LeftBound81350.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29605⟩⟩) (rawTerms := some (Proof.Events317.exact81351RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound81350.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88620.bound, LeftBound81350.bound]
def bound : CoeffClass := .finite ⟨20673980874611694436352, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88620.bound, LeftBound81350.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88620.actual selector witness, LeftBound81350.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88625

namespace LeftBound88629
def owner : Owner := ⟨.program ⟨214⟩, ⟨29823⟩⟩
def transferEvent : Nat := 88629
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 88627 .coefficient, .predecessor 1 88628 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88627 .coefficient)
      LeftBound88624.bound (LeftBound88624.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events346.exact88626RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88624.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88624.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88628 .coefficient)
      LeftBound80867.bound (LeftBound80867.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events315.exact80871RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80867.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80867.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88624.bound, LeftBound80867.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88624.bound, LeftBound80867.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88624.actual selector witness, LeftBound80867.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88629

namespace LeftBound88630
def owner : Owner := ⟨.program ⟨214⟩, ⟨29823⟩⟩
def transferEvent : Nat := 88630
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 88626 .summary, .result 80871 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 88626 .summary)
      LeftBound88625.bound (LeftBound88625.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29606⟩⟩) (rawTerms := some (Proof.Events346.exact88626RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound88625.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80871 .summary)
      LeftBound80870.bound (LeftBound80870.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29822⟩⟩) (rawTerms := some (Proof.Events315.exact80871RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80870.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88625.bound, LeftBound80870.bound]
def bound : CoeffClass := .finite ⟨21966497597451692486656, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88625.bound, LeftBound80870.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88625.actual selector witness, LeftBound80870.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88630

namespace LeftBound88634
def owner : Owner := ⟨.program ⟨214⟩, ⟨30120⟩⟩
def transferEvent : Nat := 88634
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 88632 .coefficient, .predecessor 1 88633 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88632 .coefficient)
      LeftBound88629.bound (LeftBound88629.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events346.exact88631RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88629.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88629.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88633 .coefficient)
      LeftBound80387.bound (LeftBound80387.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events314.exact80391RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80387.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80387.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88629.bound, LeftBound80387.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88629.bound, LeftBound80387.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88629.actual selector witness, LeftBound80387.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88634

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
