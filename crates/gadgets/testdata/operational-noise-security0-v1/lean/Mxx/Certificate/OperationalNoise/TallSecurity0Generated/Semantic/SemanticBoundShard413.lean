import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard018
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard352
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard412

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound61553
def owner : Owner := ⟨.program ⟨214⟩, ⟨29609⟩⟩
def transferEvent : Nat := 61553
def frameStart : Nat := 61476
def rule : BoundRule := .product (.predecessor 0 61551 .coefficient) (.predecessor 1 61552 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61551 .coefficient)
      LeftBound61549.bound (LeftBound61549.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events240.exact61550RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61549.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61549.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61552 .coefficient)
      LeftAuthority61526.bound (LeftAuthority61526.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events240.exact61527RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61526.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61526.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound61549.bound LeftAuthority61526.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61549.bound, LeftAuthority61526.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound61549.actual selector witness) * (LeftAuthority61526.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound61553

namespace LeftBound61564
def owner : Owner := ⟨.program ⟨214⟩, ⟨17500⟩⟩
def transferEvent : Nat := 61564
def frameStart : Nat := 61476
def rule : BoundRule := .product (.predecessor 0 61562 .coefficient) (.predecessor 1 61563 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61562 .coefficient)
      LeftAuthority61537.bound (LeftAuthority61537.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events240.exact61538RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61537.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61537.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61563 .coefficient)
      LeftAuthority61560.bound (LeftAuthority61560.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events240.exact61561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61560.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61560.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority61537.bound LeftAuthority61560.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority61537.bound, LeftAuthority61560.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority61537.actual selector witness) * (LeftAuthority61560.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound61564

namespace LeftBound61572
def owner : Owner := ⟨.program ⟨214⟩, ⟨17501⟩⟩
def transferEvent : Nat := 61572
def frameStart : Nat := 61476
def rule : BoundRule := .sum [.predecessor 0 61570 .coefficient, .predecessor 1 61571 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61570 .coefficient)
      LeftAuthority61568.bound (LeftAuthority61568.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events240.exact61569RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61568.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61568.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61571 .coefficient)
      LeftBound61564.bound (LeftBound61564.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events240.exact61566RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61564.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61564.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority61568.bound, LeftBound61564.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority61568.bound, LeftBound61564.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority61568.actual selector witness, LeftBound61564.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound61572

namespace LeftBound61576
def owner : Owner := ⟨.program ⟨214⟩, ⟨29614⟩⟩
def transferEvent : Nat := 61576
def frameStart : Nat := 61476
def rule : BoundRule := .sum [.predecessor 0 61574 .coefficient, .predecessor 1 61575 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61574 .coefficient)
      LeftBound61572.bound (LeftBound61572.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events240.exact61573RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61572.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61572.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61575 .coefficient)
      LeftBound61553.bound (LeftBound61553.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events240.exact61558RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61553.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61553.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound61572.bound, LeftBound61553.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61572.bound, LeftBound61553.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound61572.actual selector witness, LeftBound61553.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound61576

namespace LeftBound61589
def owner : Owner := ⟨.program ⟨214⟩, ⟨29611⟩⟩
def transferEvent : Nat := 61589
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 61587 .coefficient, .predecessor 1 61588 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61587 .coefficient)
      LeftBound61418.bound (LeftBound61418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events240.exact61586RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61418.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61588 .coefficient)
      LeftBound61401.bound (LeftBound61401.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events239.exact61408RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61401.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61401.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound61418.bound, LeftBound61401.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61418.bound, LeftBound61401.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound61418.actual selector witness, LeftBound61401.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound61589

namespace LeftBound61592
def owner : Owner := ⟨.program ⟨214⟩, ⟨29611⟩⟩
def transferEvent : Nat := 61592
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 61586 .summary, .result 61408 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 61586 .summary)
      LeftBound61420.bound (LeftBound61420.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22487⟩⟩) (rawTerms := some (Proof.Events240.exact61586RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound61420.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 61408 .summary)
      LeftBound61403.bound (LeftBound61403.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29610⟩⟩) (rawTerms := some (Proof.Events239.exact61408RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound61403.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound61420.bound, LeftBound61403.bound]
def bound : CoeffClass := .finite ⟨1292449485504936292352, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61420.bound, LeftBound61403.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound61420.actual selector witness, LeftBound61403.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound61592

namespace LeftBound61596
def owner : Owner := ⟨.program ⟨214⟩, ⟨29612⟩⟩
def transferEvent : Nat := 61596
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 61594 .coefficient) (.predecessor 1 61595 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61594 .coefficient)
      LeftBound61589.bound (LeftBound61589.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events240.exact61593RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61589.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61589.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61595 .coefficient)
      LeftBound5558.bound (LeftBound5558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5559RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5558.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound61589.bound LeftBound5558.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61589.bound, LeftBound5558.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound61589.actual selector witness) * (LeftBound5558.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound61596

namespace LeftBound61597
def owner : Owner := ⟨.program ⟨214⟩, ⟨29612⟩⟩
def transferEvent : Nat := 61597
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6661⟩⟩]⟩ [⟨.result 5555 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5555 .coefficient)
      LeftAuthority5554.bound (LeftAuthority5554.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6661⟩⟩) (rawTerms := some (Proof.Events021.exact5555RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5554.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5554.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5554.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5554.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5554.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound61597

namespace LeftBound61598
def owner : Owner := ⟨.program ⟨214⟩, ⟨29612⟩⟩
def transferEvent : Nat := 61598
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 61593 .summary) (.transfer 61597) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 61593 .summary)
      LeftBound61592.bound (LeftBound61592.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29611⟩⟩) (rawTerms := some (Proof.Events240.exact61593RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound61592.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 61597)
      LeftBound61597.bound (LeftBound61597.actual selector witness) := by
  exact .transfer (LeftBound61597.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound61592.bound LeftBound61597.bound
def bound : CoeffClass := .finite ⟨4743310290994884271912517632, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61592.bound, LeftBound61597.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound61592.actual selector witness) * (LeftBound61597.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound61598

namespace LeftBound61613
def owner : Owner := ⟨.program ⟨214⟩, ⟨29393⟩⟩
def transferEvent : Nat := 61613
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 61611 .coefficient) (.predecessor 1 61612 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61611 .coefficient)
      LeftBound52390.bound (LeftBound52390.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events204.exact52394RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52390.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52390.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61612 .coefficient)
      LeftAuthority61609.bound (LeftAuthority61609.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events240.exact61610RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61609.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61609.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound52390.bound LeftAuthority61609.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52390.bound, LeftAuthority61609.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound52390.actual selector witness) * (LeftAuthority61609.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound61613

namespace LeftBound61614
def owner : Owner := ⟨.program ⟨214⟩, ⟨29393⟩⟩
def transferEvent : Nat := 61614
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29391⟩⟩]⟩ [⟨.result 61610 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 61610 .coefficient)
      LeftAuthority61609.bound (LeftAuthority61609.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29391⟩⟩) (rawTerms := some (Proof.Events240.exact61610RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61609.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61609.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority61609.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority61609.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority61609.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound61614

namespace LeftBound61615
def owner : Owner := ⟨.program ⟨214⟩, ⟨29393⟩⟩
def transferEvent : Nat := 61615
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 52394 .summary) (.transfer 61614) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 52394 .summary)
      LeftBound52393.bound (LeftBound52393.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25534⟩⟩) (rawTerms := some (Proof.Events204.exact52394RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound52393.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 61614)
      LeftBound61614.bound (LeftBound61614.actual selector witness) := by
  exact .transfer (LeftBound61614.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound52393.bound LeftBound61614.bound
def bound : CoeffClass := .finite ⟨1292382246358571024384, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52393.bound, LeftBound61614.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound52393.actual selector witness) * (LeftBound61614.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound61615

namespace LeftBound61626
def owner : Owner := ⟨.program ⟨214⟩, ⟨22342⟩⟩
def transferEvent : Nat := 61626
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 61624 .coefficient) (.value (.predecessor 1 61625 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61624 .coefficient)
      LeftAuthority61622.bound (LeftAuthority61622.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events240.exact61623RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61622.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61622.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61625 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority61622.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority61622.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority61622.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound61626

namespace LeftBound61630
def owner : Owner := ⟨.program ⟨214⟩, ⟨22343⟩⟩
def transferEvent : Nat := 61630
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 61628 .coefficient) (.predecessor 1 61629 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61628 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61629 .coefficient)
      LeftBound61626.bound (LeftBound61626.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events240.exact61627RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61626.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61626.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound61626.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound61626.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound61626.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound61630

namespace LeftBound61631
def owner : Owner := ⟨.program ⟨214⟩, ⟨22343⟩⟩
def transferEvent : Nat := 61631
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22340⟩⟩]⟩ [⟨.result 61623 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 61623 .coefficient)
      LeftAuthority61622.bound (LeftAuthority61622.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22340⟩⟩) (rawTerms := some (Proof.Events240.exact61623RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61622.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61622.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority61622.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority61622.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority61622.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound61631

namespace LeftBound61632
def owner : Owner := ⟨.program ⟨214⟩, ⟨22343⟩⟩
def transferEvent : Nat := 61632
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 61631) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 61631)
      LeftBound61631.bound (LeftBound61631.actual selector witness) := by
  exact .transfer (LeftBound61631.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound61631.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound61631.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound61631.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound61632

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
