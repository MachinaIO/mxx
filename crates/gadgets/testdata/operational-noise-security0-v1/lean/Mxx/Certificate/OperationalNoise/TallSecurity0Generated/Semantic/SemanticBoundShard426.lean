import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard388
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard425

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound63661
def owner : Owner := ⟨.program ⟨214⟩, ⟨15783⟩⟩
def transferEvent : Nat := 63661
def frameStart : Nat := 63596
def rule : BoundRule := .product (.predecessor 0 63659 .coefficient) (.predecessor 1 63660 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63659 .coefficient)
      LeftAuthority63657.bound (LeftAuthority63657.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events248.exact63658RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63657.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63657.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63660 .coefficient)
      LeftBound63655.bound (LeftBound63655.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events248.exact63656RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63655.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63655.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority63657.bound LeftBound63655.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63657.bound, LeftBound63655.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority63657.actual selector witness) * (LeftBound63655.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63661

namespace LeftBound63669
def owner : Owner := ⟨.program ⟨214⟩, ⟨15784⟩⟩
def transferEvent : Nat := 63669
def frameStart : Nat := 63596
def rule : BoundRule := .sum [.predecessor 0 63667 .coefficient, .predecessor 1 63668 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63667 .coefficient)
      LeftAuthority63665.bound (LeftAuthority63665.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events248.exact63666RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63665.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63665.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63668 .coefficient)
      LeftBound63661.bound (LeftBound63661.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events248.exact63663RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63661.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63661.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority63665.bound, LeftBound63661.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63665.bound, LeftBound63661.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority63665.actual selector witness, LeftBound63661.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound63669

namespace LeftBound63673
def owner : Owner := ⟨.program ⟨214⟩, ⟨27439⟩⟩
def transferEvent : Nat := 63673
def frameStart : Nat := 63596
def rule : BoundRule := .product (.predecessor 0 63671 .coefficient) (.predecessor 1 63672 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63671 .coefficient)
      LeftBound63669.bound (LeftBound63669.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events248.exact63670RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63669.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63669.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63672 .coefficient)
      LeftAuthority63646.bound (LeftAuthority63646.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events248.exact63647RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63646.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63646.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound63669.bound LeftAuthority63646.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63669.bound, LeftAuthority63646.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound63669.actual selector witness) * (LeftAuthority63646.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63673

namespace LeftBound63684
def owner : Owner := ⟨.program ⟨214⟩, ⟨17444⟩⟩
def transferEvent : Nat := 63684
def frameStart : Nat := 63596
def rule : BoundRule := .product (.predecessor 0 63682 .coefficient) (.predecessor 1 63683 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63682 .coefficient)
      LeftAuthority63657.bound (LeftAuthority63657.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events248.exact63658RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63657.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63657.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63683 .coefficient)
      LeftAuthority63680.bound (LeftAuthority63680.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events248.exact63681RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63680.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63680.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority63657.bound LeftAuthority63680.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63657.bound, LeftAuthority63680.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority63657.actual selector witness) * (LeftAuthority63680.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63684

namespace LeftBound63692
def owner : Owner := ⟨.program ⟨214⟩, ⟨17445⟩⟩
def transferEvent : Nat := 63692
def frameStart : Nat := 63596
def rule : BoundRule := .sum [.predecessor 0 63690 .coefficient, .predecessor 1 63691 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63690 .coefficient)
      LeftAuthority63688.bound (LeftAuthority63688.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events248.exact63689RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63688.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63688.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63691 .coefficient)
      LeftBound63684.bound (LeftBound63684.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events248.exact63686RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63684.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63684.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority63688.bound, LeftBound63684.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63688.bound, LeftBound63684.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority63688.actual selector witness, LeftBound63684.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound63692

namespace LeftBound63696
def owner : Owner := ⟨.program ⟨214⟩, ⟨27444⟩⟩
def transferEvent : Nat := 63696
def frameStart : Nat := 63596
def rule : BoundRule := .sum [.predecessor 0 63694 .coefficient, .predecessor 1 63695 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63694 .coefficient)
      LeftBound63692.bound (LeftBound63692.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events248.exact63693RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63692.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63692.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63695 .coefficient)
      LeftBound63673.bound (LeftBound63673.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events248.exact63678RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63673.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63673.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound63692.bound, LeftBound63673.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63692.bound, LeftBound63673.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound63692.actual selector witness, LeftBound63673.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound63696

namespace LeftBound63709
def owner : Owner := ⟨.program ⟨214⟩, ⟨27441⟩⟩
def transferEvent : Nat := 63709
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 63707 .coefficient, .predecessor 1 63708 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63707 .coefficient)
      LeftBound63538.bound (LeftBound63538.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events248.exact63706RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63538.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63538.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63708 .coefficient)
      LeftBound63521.bound (LeftBound63521.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events248.exact63528RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63521.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63521.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound63538.bound, LeftBound63521.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63538.bound, LeftBound63521.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound63538.actual selector witness, LeftBound63521.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound63709

namespace LeftBound63712
def owner : Owner := ⟨.program ⟨214⟩, ⟨27441⟩⟩
def transferEvent : Nat := 63712
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 63706 .summary, .result 63528 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 63706 .summary)
      LeftBound63540.bound (LeftBound63540.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21047⟩⟩) (rawTerms := some (Proof.Events248.exact63706RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound63540.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 63528 .summary)
      LeftBound63523.bound (LeftBound63523.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27440⟩⟩) (rawTerms := some (Proof.Events248.exact63528RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound63523.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound63540.bound, LeftBound63523.bound]
def bound : CoeffClass := .finite ⟨1292001236604524572672, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63540.bound, LeftBound63523.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound63540.actual selector witness, LeftBound63523.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound63712

namespace LeftBound63716
def owner : Owner := ⟨.program ⟨214⟩, ⟨27442⟩⟩
def transferEvent : Nat := 63716
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 63714 .coefficient) (.predecessor 1 63715 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63714 .coefficient)
      LeftBound63709.bound (LeftBound63709.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events248.exact63713RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63709.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63709.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63715 .coefficient)
      LeftBound5758.bound (LeftBound5758.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5759RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5758.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5758.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound63709.bound LeftBound5758.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63709.bound, LeftBound5758.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound63709.actual selector witness) * (LeftBound5758.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63716

namespace LeftBound63717
def owner : Owner := ⟨.program ⟨214⟩, ⟨27442⟩⟩
def transferEvent : Nat := 63717
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩ [⟨.result 5755 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5755 .coefficient)
      LeftAuthority5754.bound (LeftAuthority5754.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6647⟩⟩) (rawTerms := some (Proof.Events022.exact5755RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5754.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5754.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5754.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5754.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5754.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound63717

namespace LeftBound63718
def owner : Owner := ⟨.program ⟨214⟩, ⟨27442⟩⟩
def transferEvent : Nat := 63718
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 63713 .summary) (.transfer 63717) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 63713 .summary)
      LeftBound63712.bound (LeftBound63712.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27441⟩⟩) (rawTerms := some (Proof.Events248.exact63713RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound63712.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 63717)
      LeftBound63717.bound (LeftBound63717.actual selector witness) := by
  exact .transfer (LeftBound63717.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound63712.bound LeftBound63717.bound
def bound : CoeffClass := .finite ⟨4741665210358390854099402752, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63712.bound, LeftBound63717.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound63712.actual selector witness) * (LeftBound63717.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63718

namespace LeftBound63733
def owner : Owner := ⟨.program ⟨214⟩, ⟨27223⟩⟩
def transferEvent : Nat := 63733
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 63731 .coefficient) (.predecessor 1 63732 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63731 .coefficient)
      LeftBound57210.bound (LeftBound57210.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events223.exact57214RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57210.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57210.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63732 .coefficient)
      LeftAuthority63729.bound (LeftAuthority63729.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events248.exact63730RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63729.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63729.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound57210.bound LeftAuthority63729.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57210.bound, LeftAuthority63729.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound57210.actual selector witness) * (LeftAuthority63729.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63733

namespace LeftBound63734
def owner : Owner := ⟨.program ⟨214⟩, ⟨27223⟩⟩
def transferEvent : Nat := 63734
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27221⟩⟩]⟩ [⟨.result 63730 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 63730 .coefficient)
      LeftAuthority63729.bound (LeftAuthority63729.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27221⟩⟩) (rawTerms := some (Proof.Events248.exact63730RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63729.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63729.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority63729.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63729.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority63729.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound63734

namespace LeftBound63735
def owner : Owner := ⟨.program ⟨214⟩, ⟨27223⟩⟩
def transferEvent : Nat := 63735
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 57214 .summary) (.transfer 63734) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 57214 .summary)
      LeftBound57213.bound (LeftBound57213.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25842⟩⟩) (rawTerms := some (Proof.Events223.exact57214RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound57213.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 63734)
      LeftBound63734.bound (LeftBound63734.actual selector witness) := by
  exact .transfer (LeftBound63734.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound57213.bound LeftBound63734.bound
def bound : CoeffClass := .finite ⟨1291978822348200476672, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57213.bound, LeftBound63734.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound57213.actual selector witness) * (LeftBound63734.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63735

namespace LeftBound63746
def owner : Owner := ⟨.program ⟨214⟩, ⟨20902⟩⟩
def transferEvent : Nat := 63746
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 63744 .coefficient) (.value (.predecessor 1 63745 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63744 .coefficient)
      LeftAuthority63742.bound (LeftAuthority63742.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events248.exact63743RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63742.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63742.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63745 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority63742.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63742.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority63742.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound63746

namespace LeftBound63750
def owner : Owner := ⟨.program ⟨214⟩, ⟨20903⟩⟩
def transferEvent : Nat := 63750
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 63748 .coefficient) (.predecessor 1 63749 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63748 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63749 .coefficient)
      LeftBound63746.bound (LeftBound63746.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events249.exact63747RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63746.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63746.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound63746.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound63746.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound63746.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63750

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
