import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard384
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard424

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound63484
def owner : Owner := ⟨.program ⟨214⟩, ⟨27661⟩⟩
def transferEvent : Nat := 63484
def frameStart : Nat := 63384
def rule : BoundRule := .sum [.predecessor 0 63482 .coefficient, .predecessor 1 63483 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63482 .coefficient)
      LeftBound63480.bound (LeftBound63480.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63481RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63480.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63480.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63483 .coefficient)
      LeftBound63461.bound (LeftBound63461.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63466RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63461.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63461.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound63480.bound, LeftBound63461.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63480.bound, LeftBound63461.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound63480.actual selector witness, LeftBound63461.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound63484

namespace LeftBound63497
def owner : Owner := ⟨.program ⟨214⟩, ⟨27658⟩⟩
def transferEvent : Nat := 63497
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 63495 .coefficient, .predecessor 1 63496 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63495 .coefficient)
      LeftBound63326.bound (LeftBound63326.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events248.exact63494RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63326.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63326.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63496 .coefficient)
      LeftBound63309.bound (LeftBound63309.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63316RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63309.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63309.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound63326.bound, LeftBound63309.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63326.bound, LeftBound63309.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound63326.actual selector witness, LeftBound63309.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound63497

namespace LeftBound63500
def owner : Owner := ⟨.program ⟨214⟩, ⟨27658⟩⟩
def transferEvent : Nat := 63500
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 63494 .summary, .result 63316 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 63494 .summary)
      LeftBound63328.bound (LeftBound63328.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21191⟩⟩) (rawTerms := some (Proof.Events248.exact63494RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound63328.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 63316 .summary)
      LeftBound63311.bound (LeftBound63311.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27657⟩⟩) (rawTerms := some (Proof.Events247.exact63316RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound63311.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound63328.bound, LeftBound63311.bound]
def bound : CoeffClass := .finite ⟨1292046061494565744640, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63328.bound, LeftBound63311.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound63328.actual selector witness, LeftBound63311.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound63500

namespace LeftBound63504
def owner : Owner := ⟨.program ⟨214⟩, ⟨27659⟩⟩
def transferEvent : Nat := 63504
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 63502 .coefficient) (.predecessor 1 63503 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63502 .coefficient)
      LeftBound63497.bound (LeftBound63497.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events248.exact63501RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63497.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63497.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63503 .coefficient)
      LeftBound5738.bound (LeftBound5738.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5739RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5738.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5738.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound63497.bound LeftBound5738.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63497.bound, LeftBound5738.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound63497.actual selector witness) * (LeftBound5738.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63504

namespace LeftBound63505
def owner : Owner := ⟨.program ⟨214⟩, ⟨27659⟩⟩
def transferEvent : Nat := 63505
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩ [⟨.result 5735 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5735 .coefficient)
      LeftAuthority5734.bound (LeftAuthority5734.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6643⟩⟩) (rawTerms := some (Proof.Events022.exact5735RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5734.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5734.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5734.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5734.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5734.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound63505

namespace LeftBound63506
def owner : Owner := ⟨.program ⟨214⟩, ⟨27659⟩⟩
def transferEvent : Nat := 63506
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 63501 .summary) (.transfer 63505) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 63501 .summary)
      LeftBound63500.bound (LeftBound63500.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27658⟩⟩) (rawTerms := some (Proof.Events248.exact63501RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound63500.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 63505)
      LeftBound63505.bound (LeftBound63505.actual selector witness) := by
  exact .transfer (LeftBound63505.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound63500.bound LeftBound63505.bound
def bound : CoeffClass := .finite ⟨4741829718422040195880714240, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63500.bound, LeftBound63505.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound63500.actual selector witness) * (LeftBound63505.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63506

namespace LeftBound63521
def owner : Owner := ⟨.program ⟨214⟩, ⟨27440⟩⟩
def transferEvent : Nat := 63521
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 63519 .coefficient) (.predecessor 1 63520 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63519 .coefficient)
      LeftBound56728.bound (LeftBound56728.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56732RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56728.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56728.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63520 .coefficient)
      LeftAuthority63517.bound (LeftAuthority63517.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events248.exact63518RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63517.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63517.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound56728.bound LeftAuthority63517.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56728.bound, LeftAuthority63517.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound56728.actual selector witness) * (LeftAuthority63517.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63521

namespace LeftBound63522
def owner : Owner := ⟨.program ⟨214⟩, ⟨27440⟩⟩
def transferEvent : Nat := 63522
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27438⟩⟩]⟩ [⟨.result 63518 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 63518 .coefficient)
      LeftAuthority63517.bound (LeftAuthority63517.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27438⟩⟩) (rawTerms := some (Proof.Events248.exact63518RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63517.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63517.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority63517.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63517.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority63517.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound63522

namespace LeftBound63523
def owner : Owner := ⟨.program ⟨214⟩, ⟨27440⟩⟩
def transferEvent : Nat := 63523
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 56732 .summary) (.transfer 63522) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 56732 .summary)
      LeftBound56731.bound (LeftBound56731.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25919⟩⟩) (rawTerms := some (Proof.Events221.exact56732RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound56731.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 63522)
      LeftBound63522.bound (LeftBound63522.actual selector witness) := by
  exact .transfer (LeftBound63522.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound56731.bound LeftBound63522.bound
def bound : CoeffClass := .finite ⟨1292001234793221062656, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56731.bound, LeftBound63522.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound56731.actual selector witness) * (LeftBound63522.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63523

namespace LeftBound63534
def owner : Owner := ⟨.program ⟨214⟩, ⟨21046⟩⟩
def transferEvent : Nat := 63534
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 63532 .coefficient) (.value (.predecessor 1 63533 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63532 .coefficient)
      LeftAuthority63530.bound (LeftAuthority63530.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events248.exact63531RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63530.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63530.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63533 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority63530.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63530.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority63530.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound63534

namespace LeftBound63538
def owner : Owner := ⟨.program ⟨214⟩, ⟨21047⟩⟩
def transferEvent : Nat := 63538
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 63536 .coefficient) (.predecessor 1 63537 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63536 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63537 .coefficient)
      LeftBound63534.bound (LeftBound63534.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events248.exact63535RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63534.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63534.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound63534.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound63534.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound63534.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63538

namespace LeftBound63539
def owner : Owner := ⟨.program ⟨214⟩, ⟨21047⟩⟩
def transferEvent : Nat := 63539
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21044⟩⟩]⟩ [⟨.result 63531 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 63531 .coefficient)
      LeftAuthority63530.bound (LeftAuthority63530.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21044⟩⟩) (rawTerms := some (Proof.Events248.exact63531RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63530.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63530.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority63530.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63530.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority63530.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound63539

namespace LeftBound63540
def owner : Owner := ⟨.program ⟨214⟩, ⟨21047⟩⟩
def transferEvent : Nat := 63540
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 63539) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 63539)
      LeftBound63539.bound (LeftBound63539.actual selector witness) := by
  exact .transfer (LeftBound63539.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound63539.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound63539.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound63539.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63540

namespace LeftBound63635
def owner : Owner := ⟨.program ⟨214⟩, ⟨15707⟩⟩
def transferEvent : Nat := 63635
def frameStart : Nat := 63596
def rule : BoundRule := .identity (.predecessor 0 63634 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63634 .coefficient)
      LeftAuthority63632.bound (LeftAuthority63632.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events248.exact63633RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63632.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63632.derived selector witness)

def rawBound : CoeffClass := LeftAuthority63632.bound
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63632.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority63632.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound63635

namespace LeftBound63652
def owner : Owner := ⟨.program ⟨214⟩, ⟨15781⟩⟩
def transferEvent : Nat := 63652
def frameStart : Nat := 63596
def rule : BoundRule := .sum [.predecessor 0 63650 .coefficient, .predecessor 1 63651 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63650 .coefficient)
      LeftBound63635.bound (LeftBound63635.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound63635.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63651 .coefficient)
      LeftAuthority63648.bound (LeftAuthority63648.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority63648.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound63635.bound, LeftAuthority63648.bound]
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63635.bound, LeftAuthority63648.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound63635.actual selector witness, LeftAuthority63648.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound63652

namespace LeftBound63655
def owner : Owner := ⟨.program ⟨214⟩, ⟨15782⟩⟩
def transferEvent : Nat := 63655
def frameStart : Nat := 63596
def rule : BoundRule := .identity (.predecessor 0 63654 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63654 .coefficient)
      LeftBound63652.bound (LeftBound63652.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound63652.derived selector witness)

def rawBound : CoeffClass := LeftBound63652.bound
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63652.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound63652.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound63655

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
