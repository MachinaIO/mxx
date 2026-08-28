import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard390

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound57514
def owner : Owner := ⟨.program ⟨214⟩, ⟨19247⟩⟩
def transferEvent : Nat := 57514
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19244⟩⟩]⟩ [⟨.result 57506 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 57506 .coefficient)
      LeftAuthority57505.bound (LeftAuthority57505.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19244⟩⟩) (rawTerms := some (Proof.Events224.exact57506RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57505.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57505.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority57505.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority57505.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority57505.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound57514

namespace LeftBound57515
def owner : Owner := ⟨.program ⟨214⟩, ⟨19247⟩⟩
def transferEvent : Nat := 57515
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 57514) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 57514)
      LeftBound57514.bound (LeftBound57514.actual selector witness) := by
  exact .transfer (LeftBound57514.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound57514.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound57514.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound57514.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57515

namespace LeftBound57594
def owner : Owner := ⟨.program ⟨214⟩, ⟨12173⟩⟩
def transferEvent : Nat := 57594
def frameStart : Nat := 57565
def rule : BoundRule := .product (.predecessor 0 57592 .coefficient) (.predecessor 1 57593 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57592 .coefficient)
      LeftAuthority57590.bound (LeftAuthority57590.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events224.exact57591RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57590.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57590.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57593 .coefficient)
      LeftAuthority57587.bound (LeftAuthority57587.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events224.exact57588RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57587.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57587.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority57590.bound LeftAuthority57587.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority57590.bound, LeftAuthority57587.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority57590.actual selector witness) * (LeftAuthority57587.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57594

namespace LeftBound57598
def owner : Owner := ⟨.program ⟨214⟩, ⟨12174⟩⟩
def transferEvent : Nat := 57598
def frameStart : Nat := 57565
def rule : BoundRule := .identity (.predecessor 0 57597 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57597 .coefficient)
      LeftBound57594.bound (LeftBound57594.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events224.exact57596RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57594.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57594.derived selector witness)

def rawBound : CoeffClass := LeftBound57594.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57594.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound57594.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound57598

namespace LeftBound57615
def owner : Owner := ⟨.program ⟨214⟩, ⟨12274⟩⟩
def transferEvent : Nat := 57615
def frameStart : Nat := 57565
def rule : BoundRule := .sum [.predecessor 0 57613 .coefficient, .predecessor 1 57614 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57613 .coefficient)
      LeftBound57598.bound (LeftBound57598.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound57598.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57614 .coefficient)
      LeftAuthority57611.bound (LeftAuthority57611.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority57611.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound57598.bound, LeftAuthority57611.bound]
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57598.bound, LeftAuthority57611.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound57598.actual selector witness, LeftAuthority57611.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound57615

namespace LeftBound57618
def owner : Owner := ⟨.program ⟨214⟩, ⟨12275⟩⟩
def transferEvent : Nat := 57618
def frameStart : Nat := 57565
def rule : BoundRule := .identity (.predecessor 0 57617 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57617 .coefficient)
      LeftBound57615.bound (LeftBound57615.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound57615.derived selector witness)

def rawBound : CoeffClass := LeftBound57615.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57615.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound57615.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound57618

namespace LeftBound57624
def owner : Owner := ⟨.program ⟨214⟩, ⟨12276⟩⟩
def transferEvent : Nat := 57624
def frameStart : Nat := 57565
def rule : BoundRule := .product (.predecessor 0 57622 .coefficient) (.predecessor 1 57623 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57622 .coefficient)
      LeftAuthority57620.bound (LeftAuthority57620.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events225.exact57621RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57620.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57620.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57623 .coefficient)
      LeftBound57618.bound (LeftBound57618.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events225.exact57619RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57618.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57618.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority57620.bound LeftBound57618.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority57620.bound, LeftBound57618.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority57620.actual selector witness) * (LeftBound57618.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57624

namespace LeftBound57640
def owner : Owner := ⟨.program ⟨214⟩, ⟨7841⟩⟩
def transferEvent : Nat := 57640
def frameStart : Nat := 57565
def rule : BoundRule := .scale (.predecessor 0 57638 .coefficient) (.value (.predecessor 1 57639 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57638 .coefficient)
      LeftAuthority57636.bound (LeftAuthority57636.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events225.exact57637RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57636.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57636.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57639 .coefficient)
      LeftAuthority57627.bound (LeftAuthority57627.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority57627.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority57636.bound LeftAuthority57627.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority57636.bound, LeftAuthority57627.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority57636.actual selector witness) * (LeftAuthority57627.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound57640

namespace LeftBound57643
def owner : Owner := ⟨.program ⟨214⟩, ⟨6792⟩⟩
def transferEvent : Nat := 57643
def frameStart : Nat := 57565
def rule : BoundRule := .identity (.predecessor 0 57642 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57642 .coefficient)
      LeftAuthority57630.bound (LeftAuthority57630.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events225.exact57631RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57630.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57630.derived selector witness)

def rawBound : CoeffClass := LeftAuthority57630.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority57630.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority57630.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound57643

namespace LeftBound57647
def owner : Owner := ⟨.program ⟨214⟩, ⟨7842⟩⟩
def transferEvent : Nat := 57647
def frameStart : Nat := 57565
def rule : BoundRule := .product (.predecessor 0 57645 .coefficient) (.predecessor 1 57646 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57645 .coefficient)
      LeftBound57643.bound (LeftBound57643.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events225.exact57644RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57643.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57643.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57646 .coefficient)
      LeftBound57640.bound (LeftBound57640.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events225.exact57641RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57640.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57640.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound57643.bound LeftBound57640.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57643.bound, LeftBound57640.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound57643.actual selector witness) * (LeftBound57640.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57647

namespace LeftBound57652
def owner : Owner := ⟨.program ⟨214⟩, ⟨12277⟩⟩
def transferEvent : Nat := 57652
def frameStart : Nat := 57565
def rule : BoundRule := .sum [.predecessor 0 57650 .coefficient, .predecessor 1 57651 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57650 .coefficient)
      LeftBound57647.bound (LeftBound57647.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events225.exact57649RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57647.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57647.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57651 .coefficient)
      LeftBound57624.bound (LeftBound57624.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events225.exact57626RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57624.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57624.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound57647.bound, LeftBound57624.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57647.bound, LeftBound57624.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound57647.actual selector witness, LeftBound57624.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound57652

namespace LeftBound57656
def owner : Owner := ⟨.program ⟨214⟩, ⟨25304⟩⟩
def transferEvent : Nat := 57656
def frameStart : Nat := 57565
def rule : BoundRule := .product (.predecessor 0 57654 .coefficient) (.predecessor 1 57655 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57654 .coefficient)
      LeftBound57652.bound (LeftBound57652.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events225.exact57653RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57652.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57652.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57655 .coefficient)
      LeftAuthority57609.bound (LeftAuthority57609.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events225.exact57610RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57609.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57609.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound57652.bound LeftAuthority57609.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57652.bound, LeftAuthority57609.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound57652.actual selector witness) * (LeftAuthority57609.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57656

namespace LeftBound57667
def owner : Owner := ⟨.program ⟨214⟩, ⟨15428⟩⟩
def transferEvent : Nat := 57667
def frameStart : Nat := 57565
def rule : BoundRule := .product (.predecessor 0 57665 .coefficient) (.predecessor 1 57666 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57665 .coefficient)
      LeftAuthority57620.bound (LeftAuthority57620.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events225.exact57621RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57620.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57620.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57666 .coefficient)
      LeftAuthority57663.bound (LeftAuthority57663.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events225.exact57664RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57663.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57663.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority57620.bound LeftAuthority57663.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority57620.bound, LeftAuthority57663.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority57620.actual selector witness) * (LeftAuthority57663.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57667

namespace LeftBound57675
def owner : Owner := ⟨.program ⟨214⟩, ⟨15429⟩⟩
def transferEvent : Nat := 57675
def frameStart : Nat := 57565
def rule : BoundRule := .sum [.predecessor 0 57673 .coefficient, .predecessor 1 57674 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57673 .coefficient)
      LeftAuthority57671.bound (LeftAuthority57671.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events225.exact57672RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57671.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57671.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57674 .coefficient)
      LeftBound57667.bound (LeftBound57667.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events225.exact57669RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57667.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57667.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority57671.bound, LeftBound57667.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority57671.bound, LeftBound57667.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority57671.actual selector witness, LeftBound57667.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound57675

namespace LeftBound57679
def owner : Owner := ⟨.program ⟨214⟩, ⟨25305⟩⟩
def transferEvent : Nat := 57679
def frameStart : Nat := 57565
def rule : BoundRule := .sum [.predecessor 0 57677 .coefficient, .predecessor 1 57678 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57677 .coefficient)
      LeftBound57675.bound (LeftBound57675.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events225.exact57676RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57675.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57675.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57678 .coefficient)
      LeftBound57656.bound (LeftBound57656.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events225.exact57661RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57656.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57656.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound57675.bound, LeftBound57656.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57675.bound, LeftBound57656.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound57675.actual selector witness, LeftBound57656.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound57679

namespace LeftBound57692
def owner : Owner := ⟨.program ⟨214⟩, ⟨25303⟩⟩
def transferEvent : Nat := 57692
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 57690 .coefficient, .predecessor 1 57691 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57690 .coefficient)
      LeftBound57513.bound (LeftBound57513.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events225.exact57689RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57513.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57513.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57691 .coefficient)
      LeftBound57496.bound (LeftBound57496.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events224.exact57503RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57496.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57496.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound57513.bound, LeftBound57496.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57513.bound, LeftBound57496.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound57513.actual selector witness, LeftBound57496.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound57692

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
