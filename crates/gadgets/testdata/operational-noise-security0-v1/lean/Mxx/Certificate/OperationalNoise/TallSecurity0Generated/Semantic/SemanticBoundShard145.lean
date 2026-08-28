import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard144

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound22590
def owner : Owner := ⟨.program ⟨214⟩, ⟨13068⟩⟩
def transferEvent : Nat := 22590
def frameStart : Nat := 22531
def rule : BoundRule := .product (.predecessor 0 22588 .coefficient) (.predecessor 1 22589 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22588 .coefficient)
      LeftAuthority22586.bound (LeftAuthority22586.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22587RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22586.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22586.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22589 .coefficient)
      LeftBound22584.bound (LeftBound22584.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22585RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22584.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22584.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority22586.bound LeftBound22584.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority22586.bound, LeftBound22584.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority22586.actual selector witness) * (LeftBound22584.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22590

namespace LeftBound22606
def owner : Owner := ⟨.program ⟨214⟩, ⟨7877⟩⟩
def transferEvent : Nat := 22606
def frameStart : Nat := 22531
def rule : BoundRule := .scale (.predecessor 0 22604 .coefficient) (.value (.predecessor 1 22605 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22604 .coefficient)
      LeftAuthority22602.bound (LeftAuthority22602.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22603RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22602.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22602.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22605 .coefficient)
      LeftAuthority22593.bound (LeftAuthority22593.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority22593.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority22602.bound LeftAuthority22593.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority22602.bound, LeftAuthority22593.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority22602.actual selector witness) * (LeftAuthority22593.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound22606

namespace LeftBound22609
def owner : Owner := ⟨.program ⟨214⟩, ⟨6768⟩⟩
def transferEvent : Nat := 22609
def frameStart : Nat := 22531
def rule : BoundRule := .identity (.predecessor 0 22608 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22608 .coefficient)
      LeftAuthority22596.bound (LeftAuthority22596.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22597RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22596.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22596.derived selector witness)

def rawBound : CoeffClass := LeftAuthority22596.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority22596.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority22596.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound22609

namespace LeftBound22613
def owner : Owner := ⟨.program ⟨214⟩, ⟨7878⟩⟩
def transferEvent : Nat := 22613
def frameStart : Nat := 22531
def rule : BoundRule := .product (.predecessor 0 22611 .coefficient) (.predecessor 1 22612 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22611 .coefficient)
      LeftBound22609.bound (LeftBound22609.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22610RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22609.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22609.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22612 .coefficient)
      LeftBound22606.bound (LeftBound22606.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22607RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22606.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22606.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound22609.bound LeftBound22606.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22609.bound, LeftBound22606.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound22609.actual selector witness) * (LeftBound22606.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22613

namespace LeftBound22618
def owner : Owner := ⟨.program ⟨214⟩, ⟨13069⟩⟩
def transferEvent : Nat := 22618
def frameStart : Nat := 22531
def rule : BoundRule := .sum [.predecessor 0 22616 .coefficient, .predecessor 1 22617 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22616 .coefficient)
      LeftBound22613.bound (LeftBound22613.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22615RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22613.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22613.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22617 .coefficient)
      LeftBound22590.bound (LeftBound22590.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22592RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22590.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22590.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound22613.bound, LeftBound22590.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22613.bound, LeftBound22590.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound22613.actual selector witness, LeftBound22590.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound22618

namespace LeftBound22622
def owner : Owner := ⟨.program ⟨214⟩, ⟨25622⟩⟩
def transferEvent : Nat := 22622
def frameStart : Nat := 22531
def rule : BoundRule := .product (.predecessor 0 22620 .coefficient) (.predecessor 1 22621 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22620 .coefficient)
      LeftBound22618.bound (LeftBound22618.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22619RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22618.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22618.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22621 .coefficient)
      LeftAuthority22575.bound (LeftAuthority22575.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22576RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22575.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22575.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound22618.bound LeftAuthority22575.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22618.bound, LeftAuthority22575.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound22618.actual selector witness) * (LeftAuthority22575.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22622

namespace LeftBound22633
def owner : Owner := ⟨.program ⟨214⟩, ⟨16766⟩⟩
def transferEvent : Nat := 22633
def frameStart : Nat := 22531
def rule : BoundRule := .product (.predecessor 0 22631 .coefficient) (.predecessor 1 22632 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22631 .coefficient)
      LeftAuthority22586.bound (LeftAuthority22586.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22587RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22586.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22586.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22632 .coefficient)
      LeftAuthority22629.bound (LeftAuthority22629.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22630RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22629.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22629.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority22586.bound LeftAuthority22629.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority22586.bound, LeftAuthority22629.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority22586.actual selector witness) * (LeftAuthority22629.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22633

namespace LeftBound22641
def owner : Owner := ⟨.program ⟨214⟩, ⟨16767⟩⟩
def transferEvent : Nat := 22641
def frameStart : Nat := 22531
def rule : BoundRule := .sum [.predecessor 0 22639 .coefficient, .predecessor 1 22640 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22639 .coefficient)
      LeftAuthority22637.bound (LeftAuthority22637.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22638RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22637.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22637.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22640 .coefficient)
      LeftBound22633.bound (LeftBound22633.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22635RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22633.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22633.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority22637.bound, LeftBound22633.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority22637.bound, LeftBound22633.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority22637.actual selector witness, LeftBound22633.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound22641

namespace LeftBound22645
def owner : Owner := ⟨.program ⟨214⟩, ⟨25623⟩⟩
def transferEvent : Nat := 22645
def frameStart : Nat := 22531
def rule : BoundRule := .sum [.predecessor 0 22643 .coefficient, .predecessor 1 22644 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22643 .coefficient)
      LeftBound22641.bound (LeftBound22641.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22642RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22641.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22641.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22644 .coefficient)
      LeftBound22622.bound (LeftBound22622.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22627RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22622.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22622.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound22641.bound, LeftBound22622.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22641.bound, LeftBound22622.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound22641.actual selector witness, LeftBound22622.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound22645

namespace LeftBound22658
def owner : Owner := ⟨.program ⟨214⟩, ⟨25621⟩⟩
def transferEvent : Nat := 22658
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 22656 .coefficient, .predecessor 1 22657 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22656 .coefficient)
      LeftBound22479.bound (LeftBound22479.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22655RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22479.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22479.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22657 .coefficient)
      LeftBound22462.bound (LeftBound22462.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events087.exact22469RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22462.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22462.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound22479.bound, LeftBound22462.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22479.bound, LeftBound22462.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound22479.actual selector witness, LeftBound22462.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound22658

namespace LeftBound22661
def owner : Owner := ⟨.program ⟨214⟩, ⟨25621⟩⟩
def transferEvent : Nat := 22661
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 22655 .summary, .result 22469 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 22655 .summary)
      LeftBound22481.bound (LeftBound22481.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20119⟩⟩) (rawTerms := some (Proof.Events088.exact22655RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound22481.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 22469 .summary)
      LeftBound22464.bound (LeftBound22464.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25620⟩⟩) (rawTerms := some (Proof.Events087.exact22469RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound22464.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound22481.bound, LeftBound22464.bound]
def bound : CoeffClass := .finite ⟨352164536528896, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22481.bound, LeftBound22464.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound22481.actual selector witness, LeftBound22464.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound22661

namespace LeftBound22665
def owner : Owner := ⟨.program ⟨214⟩, ⟨29643⟩⟩
def transferEvent : Nat := 22665
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 22663 .coefficient) (.predecessor 1 22664 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22663 .coefficient)
      LeftBound22658.bound (LeftBound22658.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22662RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22658.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22658.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22664 .coefficient)
      LeftAuthority22384.bound (LeftAuthority22384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events087.exact22385RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22384.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22384.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound22658.bound LeftAuthority22384.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22658.bound, LeftAuthority22384.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound22658.actual selector witness) * (LeftAuthority22384.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22665

namespace LeftBound22666
def owner : Owner := ⟨.program ⟨214⟩, ⟨29643⟩⟩
def transferEvent : Nat := 22666
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29641⟩⟩]⟩ [⟨.result 22385 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 22385 .coefficient)
      LeftAuthority22384.bound (LeftAuthority22384.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29641⟩⟩) (rawTerms := some (Proof.Events087.exact22385RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22384.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22384.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority22384.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority22384.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority22384.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound22666

namespace LeftBound22667
def owner : Owner := ⟨.program ⟨214⟩, ⟨29643⟩⟩
def transferEvent : Nat := 22667
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 22662 .summary) (.transfer 22666) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 22662 .summary)
      LeftBound22661.bound (LeftBound22661.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25621⟩⟩) (rawTerms := some (Proof.Events088.exact22662RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound22661.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 22666)
      LeftBound22666.bound (LeftBound22666.actual selector witness) := by
  exact .transfer (LeftBound22666.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound22661.bound LeftBound22666.bound
def bound : CoeffClass := .finite ⟨1292449483693632782336, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22661.bound, LeftBound22666.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound22661.actual selector witness) * (LeftBound22666.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22667

namespace LeftBound22678
def owner : Owner := ⟨.program ⟨214⟩, ⟨22566⟩⟩
def transferEvent : Nat := 22678
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 22676 .coefficient) (.value (.predecessor 1 22677 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22676 .coefficient)
      LeftAuthority22674.bound (LeftAuthority22674.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22675RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22674.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22674.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22677 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority22674.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority22674.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority22674.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound22678

namespace LeftBound22682
def owner : Owner := ⟨.program ⟨214⟩, ⟨22567⟩⟩
def transferEvent : Nat := 22682
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 22680 .coefficient) (.predecessor 1 22681 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22680 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22681 .coefficient)
      LeftBound22678.bound (LeftBound22678.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22679RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22678.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22678.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound22678.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound22678.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound22678.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22682

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
