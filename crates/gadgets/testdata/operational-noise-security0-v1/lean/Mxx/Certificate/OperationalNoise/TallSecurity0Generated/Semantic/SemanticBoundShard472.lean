import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard064
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard437
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard440
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard471

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound69449
def owner : Owner := ⟨.program ⟨214⟩, ⟨21687⟩⟩
def transferEvent : Nat := 69449
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 69447 .coefficient) (.predecessor 1 69448 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69447 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69448 .coefficient)
      LeftBound69445.bound (LeftBound69445.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events271.exact69446RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69445.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69445.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound69445.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound69445.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound69445.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound69449

namespace LeftBound69450
def owner : Owner := ⟨.program ⟨214⟩, ⟨21687⟩⟩
def transferEvent : Nat := 69450
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21684⟩⟩]⟩ [⟨.result 69442 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 69442 .coefficient)
      LeftAuthority69441.bound (LeftAuthority69441.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21684⟩⟩) (rawTerms := some (Proof.Events271.exact69442RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority69441.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority69441.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority69441.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority69441.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority69441.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound69450

namespace LeftBound69451
def owner : Owner := ⟨.program ⟨214⟩, ⟨21687⟩⟩
def transferEvent : Nat := 69451
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 69450) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 69450)
      LeftBound69450.bound (LeftBound69450.actual selector witness) := by
  exact .transfer (LeftBound69450.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound69450.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound69450.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound69450.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound69451

namespace LeftBound69546
def owner : Owner := ⟨.program ⟨214⟩, ⟨16175⟩⟩
def transferEvent : Nat := 69546
def frameStart : Nat := 69507
def rule : BoundRule := .identity (.predecessor 0 69545 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69545 .coefficient)
      LeftAuthority69543.bound (LeftAuthority69543.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events271.exact69544RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority69543.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority69543.derived selector witness)

def rawBound : CoeffClass := LeftAuthority69543.bound
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority69543.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority69543.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound69546

namespace LeftBound69563
def owner : Owner := ⟨.program ⟨214⟩, ⟨16214⟩⟩
def transferEvent : Nat := 69563
def frameStart : Nat := 69507
def rule : BoundRule := .sum [.predecessor 0 69561 .coefficient, .predecessor 1 69562 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69561 .coefficient)
      LeftBound69546.bound (LeftBound69546.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound69546.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69562 .coefficient)
      LeftAuthority69559.bound (LeftAuthority69559.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority69559.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69546.bound, LeftAuthority69559.bound]
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69546.bound, LeftAuthority69559.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound69546.actual selector witness, LeftAuthority69559.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69563

namespace LeftBound69566
def owner : Owner := ⟨.program ⟨214⟩, ⟨16215⟩⟩
def transferEvent : Nat := 69566
def frameStart : Nat := 69507
def rule : BoundRule := .identity (.predecessor 0 69565 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69565 .coefficient)
      LeftBound69563.bound (LeftBound69563.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound69563.derived selector witness)

def rawBound : CoeffClass := LeftBound69563.bound
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69563.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound69563.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound69566

namespace LeftBound69572
def owner : Owner := ⟨.program ⟨214⟩, ⟨16216⟩⟩
def transferEvent : Nat := 69572
def frameStart : Nat := 69507
def rule : BoundRule := .product (.predecessor 0 69570 .coefficient) (.predecessor 1 69571 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69570 .coefficient)
      LeftAuthority69568.bound (LeftAuthority69568.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events271.exact69569RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority69568.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority69568.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69571 .coefficient)
      LeftBound69566.bound (LeftBound69566.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events271.exact69567RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69566.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69566.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority69568.bound LeftBound69566.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority69568.bound, LeftBound69566.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority69568.actual selector witness) * (LeftBound69566.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound69572

namespace LeftBound69580
def owner : Owner := ⟨.program ⟨214⟩, ⟨16217⟩⟩
def transferEvent : Nat := 69580
def frameStart : Nat := 69507
def rule : BoundRule := .sum [.predecessor 0 69578 .coefficient, .predecessor 1 69579 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69578 .coefficient)
      LeftAuthority69576.bound (LeftAuthority69576.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events271.exact69577RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority69576.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority69576.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69579 .coefficient)
      LeftBound69572.bound (LeftBound69572.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events271.exact69574RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69572.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69572.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority69576.bound, LeftBound69572.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority69576.bound, LeftBound69572.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority69576.actual selector witness, LeftBound69572.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69580

namespace LeftBound69584
def owner : Owner := ⟨.program ⟨214⟩, ⟨28288⟩⟩
def transferEvent : Nat := 69584
def frameStart : Nat := 69507
def rule : BoundRule := .product (.predecessor 0 69582 .coefficient) (.predecessor 1 69583 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69582 .coefficient)
      LeftBound69580.bound (LeftBound69580.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events271.exact69581RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69580.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69580.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69583 .coefficient)
      LeftAuthority69557.bound (LeftAuthority69557.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events271.exact69558RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority69557.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority69557.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound69580.bound LeftAuthority69557.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69580.bound, LeftAuthority69557.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound69580.actual selector witness) * (LeftAuthority69557.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound69584

namespace LeftBound69595
def owner : Owner := ⟨.program ⟨214⟩, ⟨18338⟩⟩
def transferEvent : Nat := 69595
def frameStart : Nat := 69507
def rule : BoundRule := .product (.predecessor 0 69593 .coefficient) (.predecessor 1 69594 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69593 .coefficient)
      LeftAuthority69568.bound (LeftAuthority69568.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events271.exact69569RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority69568.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority69568.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69594 .coefficient)
      LeftAuthority69591.bound (LeftAuthority69591.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events271.exact69592RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority69591.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority69591.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority69568.bound LeftAuthority69591.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority69568.bound, LeftAuthority69591.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority69568.actual selector witness) * (LeftAuthority69591.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound69595

namespace LeftBound69603
def owner : Owner := ⟨.program ⟨214⟩, ⟨18339⟩⟩
def transferEvent : Nat := 69603
def frameStart : Nat := 69507
def rule : BoundRule := .sum [.predecessor 0 69601 .coefficient, .predecessor 1 69602 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69601 .coefficient)
      LeftAuthority69599.bound (LeftAuthority69599.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events271.exact69600RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority69599.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority69599.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69602 .coefficient)
      LeftBound69595.bound (LeftBound69595.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events271.exact69597RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69595.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69595.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority69599.bound, LeftBound69595.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority69599.bound, LeftBound69595.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority69599.actual selector witness, LeftBound69595.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69603

namespace LeftBound69607
def owner : Owner := ⟨.program ⟨214⟩, ⟨28292⟩⟩
def transferEvent : Nat := 69607
def frameStart : Nat := 69507
def rule : BoundRule := .sum [.predecessor 0 69605 .coefficient, .predecessor 1 69606 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69605 .coefficient)
      LeftBound69603.bound (LeftBound69603.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events271.exact69604RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69603.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69603.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69606 .coefficient)
      LeftBound69584.bound (LeftBound69584.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events271.exact69589RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69584.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69584.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69603.bound, LeftBound69584.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69603.bound, LeftBound69584.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound69603.actual selector witness, LeftBound69584.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69607

namespace LeftBound69620
def owner : Owner := ⟨.program ⟨214⟩, ⟨28290⟩⟩
def transferEvent : Nat := 69620
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 69618 .coefficient, .predecessor 1 69619 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69618 .coefficient)
      LeftBound69449.bound (LeftBound69449.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events271.exact69617RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69449.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69449.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69619 .coefficient)
      LeftBound69432.bound (LeftBound69432.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events271.exact69439RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69432.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69432.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69449.bound, LeftBound69432.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69449.bound, LeftBound69432.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound69449.actual selector witness, LeftBound69432.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69620

namespace LeftBound69623
def owner : Owner := ⟨.program ⟨214⟩, ⟨28290⟩⟩
def transferEvent : Nat := 69623
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 69617 .summary, .result 69439 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 69617 .summary)
      LeftBound69451.bound (LeftBound69451.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21687⟩⟩) (rawTerms := some (Proof.Events271.exact69617RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound69451.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 69439 .summary)
      LeftBound69434.bound (LeftBound69434.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28289⟩⟩) (rawTerms := some (Proof.Events271.exact69439RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound69434.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69451.bound, LeftBound69434.bound]
def bound : CoeffClass := .finite ⟨1292180536164689260544, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69451.bound, LeftBound69434.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound69451.actual selector witness, LeftBound69434.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69623

namespace LeftBound69647
def owner : Owner := ⟨.program ⟨214⟩, ⟨11550⟩⟩
def transferEvent : Nat := 69647
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 69645 .coefficient) (.predecessor 1 69646 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69645 .coefficient)
      LeftAuthority3292.bound (LeftAuthority3292.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events012.exact3293RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3292.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3292.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69646 .coefficient)
      LeftBound65293.bound (LeftBound65293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65293.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority3292.bound LeftBound65293.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3292.bound, LeftBound65293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority3292.actual selector witness) * (LeftBound65293.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound69647

namespace LeftBound69652
def owner : Owner := ⟨.program ⟨214⟩, ⟨7198⟩⟩
def transferEvent : Nat := 69652
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 69650 .coefficient) (.predecessor 1 69651 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69650 .coefficient)
      LeftBound65164.bound (LeftBound65164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65164.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69651 .coefficient)
      LeftBound10980.bound (LeftBound10980.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events042.exact10981RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10980.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10980.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound65164.bound LeftBound10980.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65164.bound, LeftBound10980.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound65164.actual selector witness) * (LeftBound10980.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound69652

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
