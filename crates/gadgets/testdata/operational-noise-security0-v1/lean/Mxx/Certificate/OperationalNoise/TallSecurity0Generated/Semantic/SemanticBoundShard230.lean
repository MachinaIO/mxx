import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard021
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard128
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard132
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard135
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard228
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard229

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound35506
def owner : Owner := ⟨.program ⟨214⟩, ⟨26393⟩⟩
def transferEvent : Nat := 35506
def frameStart : Nat := 35406
def rule : BoundRule := .sum [.predecessor 0 35504 .coefficient, .predecessor 1 35505 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35504 .coefficient)
      LeftBound35502.bound (LeftBound35502.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35503RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35502.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35502.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35505 .coefficient)
      LeftBound35483.bound (LeftBound35483.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35488RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35483.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35483.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35502.bound, LeftBound35483.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35502.bound, LeftBound35483.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35502.actual selector witness, LeftBound35483.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35506

namespace LeftBound35519
def owner : Owner := ⟨.program ⟨214⟩, ⟨26390⟩⟩
def transferEvent : Nat := 35519
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 35517 .coefficient, .predecessor 1 35518 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35517 .coefficient)
      LeftBound35348.bound (LeftBound35348.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35516RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35348.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35348.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35518 .coefficient)
      LeftBound35331.bound (LeftBound35331.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35338RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35331.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35331.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35348.bound, LeftBound35331.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35348.bound, LeftBound35331.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35348.actual selector witness, LeftBound35331.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35519

namespace LeftBound35522
def owner : Owner := ⟨.program ⟨214⟩, ⟨26390⟩⟩
def transferEvent : Nat := 35522
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 35516 .summary, .result 35338 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 35516 .summary)
      LeftBound35350.bound (LeftBound35350.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20335⟩⟩) (rawTerms := some (Proof.Events138.exact35516RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound35350.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 35338 .summary)
      LeftBound35333.bound (LeftBound35333.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26389⟩⟩) (rawTerms := some (Proof.Events138.exact35338RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound35333.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35350.bound, LeftBound35333.bound]
def bound : CoeffClass := .finite ⟨1291889174379421642752, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35350.bound, LeftBound35333.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35350.actual selector witness, LeftBound35333.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35522

namespace LeftBound35526
def owner : Owner := ⟨.program ⟨214⟩, ⟨26391⟩⟩
def transferEvent : Nat := 35526
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 35524 .coefficient) (.predecessor 1 35525 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35524 .coefficient)
      LeftBound35519.bound (LeftBound35519.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35523RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35519.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35519.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35525 .coefficient)
      LeftBound5858.bound (LeftBound5858.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5859RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5858.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5858.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound35519.bound LeftBound5858.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35519.bound, LeftBound5858.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound35519.actual selector witness) * (LeftBound5858.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound35526

namespace LeftBound35527
def owner : Owner := ⟨.program ⟨214⟩, ⟨26391⟩⟩
def transferEvent : Nat := 35527
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩ [⟨.result 5855 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5855 .coefficient)
      LeftAuthority5854.bound (LeftAuthority5854.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6679⟩⟩) (rawTerms := some (Proof.Events022.exact5855RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5854.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5854.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5854.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5854.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5854.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound35527

namespace LeftBound35528
def owner : Owner := ⟨.program ⟨214⟩, ⟨26391⟩⟩
def transferEvent : Nat := 35528
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 35523 .summary) (.transfer 35527) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 35523 .summary)
      LeftBound35522.bound (LeftBound35522.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26390⟩⟩) (rawTerms := some (Proof.Events138.exact35523RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound35522.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 35527)
      LeftBound35527.bound (LeftBound35527.actual selector witness) := by
  exact .transfer (LeftBound35527.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound35522.bound LeftBound35527.bound
def bound : CoeffClass := .finite ⟨4741253940199267499646124032, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35522.bound, LeftBound35527.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound35522.actual selector witness) * (LeftBound35527.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound35528

namespace LeftBound35536
def owner : Owner := ⟨.program ⟨214⟩, ⟨6629⟩⟩
def transferEvent : Nat := 35536
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 35534 .coefficient) (.predecessor 1 35535 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35534 .coefficient)
      LeftAuthority722.bound (LeftAuthority722.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact723RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority722.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority722.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35535 .coefficient)
      LeftBound21418.bound (LeftBound21418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21420RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21418.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority722.bound LeftBound21418.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority722.bound, LeftBound21418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority722.actual selector witness) * (LeftBound21418.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound35536

namespace LeftBound35541
def owner : Owner := ⟨.program ⟨214⟩, ⟨7330⟩⟩
def transferEvent : Nat := 35541
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 35539 .coefficient) (.predecessor 1 35540 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35539 .coefficient)
      LeftBound21289.bound (LeftBound21289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21289.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21289.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35540 .coefficient)
      LeftBound5872.bound (LeftBound5872.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5873RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5872.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5872.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound21289.bound LeftBound5872.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21289.bound, LeftBound5872.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound21289.actual selector witness) * (LeftBound5872.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound35541

namespace LeftBound35546
def owner : Owner := ⟨.program ⟨214⟩, ⟨7765⟩⟩
def transferEvent : Nat := 35546
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 35544 .coefficient, .predecessor 1 35545 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35544 .coefficient)
      LeftBound35541.bound (LeftBound35541.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35543RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35541.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35541.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35545 .coefficient)
      LeftBound35536.bound (LeftBound35536.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35538RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35536.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35536.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35541.bound, LeftBound35536.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35541.bound, LeftBound35536.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35541.actual selector witness, LeftBound35536.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35546

namespace LeftBound35550
def owner : Owner := ⟨.program ⟨214⟩, ⟨7766⟩⟩
def transferEvent : Nat := 35550
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 35548 .coefficient, .predecessor 1 35549 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35548 .coefficient)
      LeftBound35546.bound (LeftBound35546.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35547RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35546.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35546.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35549 .coefficient)
      LeftBound20907.bound (LeftBound20907.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events081.exact20908RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20907.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20907.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35546.bound, LeftBound20907.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35546.bound, LeftBound20907.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35546.actual selector witness, LeftBound20907.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35550

namespace LeftBound35551
def owner : Owner := ⟨.program ⟨214⟩, ⟨7766⟩⟩
def transferEvent : Nat := 35551
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨74⟩⟩]⟩ [⟨.result 20908 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20908 .coefficient)
      LeftBound20907.bound (LeftBound20907.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨74⟩⟩) (rawTerms := some (Proof.Events081.exact20908RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20907.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20907.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound20907.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20907.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound20907.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound35551

namespace LeftBound35556
def owner : Owner := ⟨.program ⟨214⟩, ⟨7811⟩⟩
def transferEvent : Nat := 35556
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 35554 .coefficient, .predecessor 1 35555 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35554 .coefficient)
      LeftBound35550.bound (LeftBound35550.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35553RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35550.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35550.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35555 .coefficient)
      LeftBound35550.bound (LeftBound35550.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35553RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35550.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35550.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35550.bound, LeftBound35550.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35550.bound, LeftBound35550.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35550.actual selector witness, LeftBound35550.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35556

namespace LeftBound35559
def owner : Owner := ⟨.program ⟨214⟩, ⟨7811⟩⟩
def transferEvent : Nat := 35559
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 35553 .summary, .result 35553 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 35553 .summary)
      LeftBound35551.bound (LeftBound35551.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨7766⟩⟩) (rawTerms := some (Proof.Events138.exact35553RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound35551.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 35553 .summary)
      LeftBound35551.bound (LeftBound35551.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨7766⟩⟩) (rawTerms := some (Proof.Events138.exact35553RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound35551.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35551.bound, LeftBound35551.bound]
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35551.bound, LeftBound35551.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35551.actual selector witness, LeftBound35551.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35559

namespace LeftBound35563
def owner : Owner := ⟨.program ⟨214⟩, ⟨26392⟩⟩
def transferEvent : Nat := 35563
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 35561 .coefficient, .predecessor 1 35562 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35561 .coefficient)
      LeftBound35556.bound (LeftBound35556.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35560RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35556.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35556.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35562 .coefficient)
      LeftBound35526.bound (LeftBound35526.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35533RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35526.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35526.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35556.bound, LeftBound35526.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35556.bound, LeftBound35526.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35556.actual selector witness, LeftBound35526.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35563

namespace LeftBound35564
def owner : Owner := ⟨.program ⟨214⟩, ⟨26392⟩⟩
def transferEvent : Nat := 35564
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 35560 .summary, .result 35533 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 35560 .summary)
      LeftBound35559.bound (LeftBound35559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨7811⟩⟩) (rawTerms := some (Proof.Events138.exact35560RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound35559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 35533 .summary)
      LeftBound35528.bound (LeftBound35528.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26391⟩⟩) (rawTerms := some (Proof.Events138.exact35533RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound35528.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35559.bound, LeftBound35528.bound]
def bound : CoeffClass := .finite ⟨4741253940199267499646124084, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35559.bound, LeftBound35528.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35559.actual selector witness, LeftBound35528.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35564

namespace LeftBound35568
def owner : Owner := ⟨.program ⟨214⟩, ⟨26601⟩⟩
def transferEvent : Nat := 35568
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 35566 .coefficient, .predecessor 1 35567 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35566 .coefficient)
      LeftBound35563.bound (LeftBound35563.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35565RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35563.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35563.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35567 .coefficient)
      LeftBound35314.bound (LeftBound35314.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events137.exact35321RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35314.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35314.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35563.bound, LeftBound35314.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35563.bound, LeftBound35314.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35563.actual selector witness, LeftBound35314.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35568

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
