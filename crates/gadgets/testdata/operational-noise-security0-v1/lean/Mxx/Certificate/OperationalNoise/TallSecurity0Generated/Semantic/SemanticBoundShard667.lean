import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard666

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound97504
def owner : Owner := ⟨.program ⟨214⟩, ⟨19736⟩⟩
def transferEvent : Nat := 97504
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19733⟩⟩]⟩ [⟨.result 97496 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 97496 .coefficient)
      LeftAuthority97495.bound (LeftAuthority97495.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19733⟩⟩) (rawTerms := some (Proof.Events380.exact97496RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97495.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97495.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority97495.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority97495.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority97495.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound97504

namespace LeftBound97505
def owner : Owner := ⟨.program ⟨214⟩, ⟨19736⟩⟩
def transferEvent : Nat := 97505
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 97504) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 97504)
      LeftBound97504.bound (LeftBound97504.actual selector witness) := by
  exact .transfer (LeftBound97504.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound97504.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound97504.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound97504.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97505

namespace LeftBound97560
def owner : Owner := ⟨.program ⟨214⟩, ⟨11738⟩⟩
def transferEvent : Nat := 97560
def frameStart : Nat := 97543
def rule : BoundRule := .product (.predecessor 0 97558 .coefficient) (.predecessor 1 97559 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97558 .coefficient)
      LeftAuthority97556.bound (LeftAuthority97556.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events381.exact97557RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97556.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97556.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97559 .coefficient)
      LeftAuthority97553.bound (LeftAuthority97553.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events381.exact97554RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97553.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97553.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority97556.bound LeftAuthority97553.bound
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority97556.bound, LeftAuthority97553.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority97556.actual selector witness) * (LeftAuthority97553.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97560

namespace LeftBound97564
def owner : Owner := ⟨.program ⟨214⟩, ⟨11739⟩⟩
def transferEvent : Nat := 97564
def frameStart : Nat := 97543
def rule : BoundRule := .identity (.predecessor 0 97563 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97563 .coefficient)
      LeftBound97560.bound (LeftBound97560.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events381.exact97562RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97560.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97560.derived selector witness)

def rawBound : CoeffClass := LeftBound97560.bound
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97560.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound97560.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound97564

namespace LeftBound97581
def owner : Owner := ⟨.program ⟨214⟩, ⟨11849⟩⟩
def transferEvent : Nat := 97581
def frameStart : Nat := 97543
def rule : BoundRule := .sum [.predecessor 0 97579 .coefficient, .predecessor 1 97580 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97579 .coefficient)
      LeftBound97564.bound (LeftBound97564.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound97564.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97580 .coefficient)
      LeftAuthority97577.bound (LeftAuthority97577.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority97577.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound97564.bound, LeftAuthority97577.bound]
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97564.bound, LeftAuthority97577.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound97564.actual selector witness, LeftAuthority97577.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound97581

namespace LeftBound97584
def owner : Owner := ⟨.program ⟨214⟩, ⟨11850⟩⟩
def transferEvent : Nat := 97584
def frameStart : Nat := 97543
def rule : BoundRule := .identity (.predecessor 0 97583 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97583 .coefficient)
      LeftBound97581.bound (LeftBound97581.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound97581.derived selector witness)

def rawBound : CoeffClass := LeftBound97581.bound
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97581.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound97581.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound97584

namespace LeftBound97590
def owner : Owner := ⟨.program ⟨214⟩, ⟨11851⟩⟩
def transferEvent : Nat := 97590
def frameStart : Nat := 97543
def rule : BoundRule := .product (.predecessor 0 97588 .coefficient) (.predecessor 1 97589 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97588 .coefficient)
      LeftAuthority97586.bound (LeftAuthority97586.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events381.exact97587RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97586.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97586.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97589 .coefficient)
      LeftBound97584.bound (LeftBound97584.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events381.exact97585RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97584.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97584.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority97586.bound LeftBound97584.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority97586.bound, LeftBound97584.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority97586.actual selector witness) * (LeftBound97584.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97590

namespace LeftBound97606
def owner : Owner := ⟨.program ⟨214⟩, ⟨7862⟩⟩
def transferEvent : Nat := 97606
def frameStart : Nat := 97543
def rule : BoundRule := .scale (.predecessor 0 97604 .coefficient) (.value (.predecessor 1 97605 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97604 .coefficient)
      LeftAuthority97602.bound (LeftAuthority97602.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events381.exact97603RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97602.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97602.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97605 .coefficient)
      LeftAuthority97593.bound (LeftAuthority97593.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority97593.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority97602.bound LeftAuthority97593.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority97602.bound, LeftAuthority97593.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority97602.actual selector witness) * (LeftAuthority97593.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound97606

namespace LeftBound97609
def owner : Owner := ⟨.program ⟨214⟩, ⟨6763⟩⟩
def transferEvent : Nat := 97609
def frameStart : Nat := 97543
def rule : BoundRule := .identity (.predecessor 0 97608 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97608 .coefficient)
      LeftAuthority97596.bound (LeftAuthority97596.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events381.exact97597RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97596.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97596.derived selector witness)

def rawBound : CoeffClass := LeftAuthority97596.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority97596.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority97596.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound97609

namespace LeftBound97613
def owner : Owner := ⟨.program ⟨214⟩, ⟨7863⟩⟩
def transferEvent : Nat := 97613
def frameStart : Nat := 97543
def rule : BoundRule := .product (.predecessor 0 97611 .coefficient) (.predecessor 1 97612 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97611 .coefficient)
      LeftBound97609.bound (LeftBound97609.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events381.exact97610RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97609.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97609.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97612 .coefficient)
      LeftBound97606.bound (LeftBound97606.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events381.exact97607RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97606.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97606.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound97609.bound LeftBound97606.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97609.bound, LeftBound97606.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound97609.actual selector witness) * (LeftBound97606.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97613

namespace LeftBound97618
def owner : Owner := ⟨.program ⟨214⟩, ⟨11852⟩⟩
def transferEvent : Nat := 97618
def frameStart : Nat := 97543
def rule : BoundRule := .sum [.predecessor 0 97616 .coefficient, .predecessor 1 97617 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97616 .coefficient)
      LeftBound97613.bound (LeftBound97613.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events381.exact97615RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97613.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97613.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97617 .coefficient)
      LeftBound97590.bound (LeftBound97590.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events381.exact97592RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97590.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97590.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound97613.bound, LeftBound97590.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97613.bound, LeftBound97590.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound97613.actual selector witness, LeftBound97590.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound97618

namespace LeftBound97622
def owner : Owner := ⟨.program ⟨214⟩, ⟨25132⟩⟩
def transferEvent : Nat := 97622
def frameStart : Nat := 97543
def rule : BoundRule := .product (.predecessor 0 97620 .coefficient) (.predecessor 1 97621 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97620 .coefficient)
      LeftBound97618.bound (LeftBound97618.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events381.exact97619RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97618.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97618.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97621 .coefficient)
      LeftAuthority97575.bound (LeftAuthority97575.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events381.exact97576RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97575.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97575.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound97618.bound LeftAuthority97575.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97618.bound, LeftAuthority97575.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound97618.actual selector witness) * (LeftAuthority97575.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97622

namespace LeftBound97633
def owner : Owner := ⟨.program ⟨214⟩, ⟨16254⟩⟩
def transferEvent : Nat := 97633
def frameStart : Nat := 97543
def rule : BoundRule := .product (.predecessor 0 97631 .coefficient) (.predecessor 1 97632 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97631 .coefficient)
      LeftAuthority97586.bound (LeftAuthority97586.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events381.exact97587RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97586.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97586.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97632 .coefficient)
      LeftAuthority97629.bound (LeftAuthority97629.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events381.exact97630RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97629.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97629.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority97586.bound LeftAuthority97629.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority97586.bound, LeftAuthority97629.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority97586.actual selector witness) * (LeftAuthority97629.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97633

namespace LeftBound97641
def owner : Owner := ⟨.program ⟨214⟩, ⟨16255⟩⟩
def transferEvent : Nat := 97641
def frameStart : Nat := 97543
def rule : BoundRule := .sum [.predecessor 0 97639 .coefficient, .predecessor 1 97640 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97639 .coefficient)
      LeftAuthority97637.bound (LeftAuthority97637.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events381.exact97638RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97637.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97637.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97640 .coefficient)
      LeftBound97633.bound (LeftBound97633.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events381.exact97635RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97633.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97633.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority97637.bound, LeftBound97633.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority97637.bound, LeftBound97633.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority97637.actual selector witness, LeftBound97633.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound97641

namespace LeftBound97645
def owner : Owner := ⟨.program ⟨214⟩, ⟨25133⟩⟩
def transferEvent : Nat := 97645
def frameStart : Nat := 97543
def rule : BoundRule := .sum [.predecessor 0 97643 .coefficient, .predecessor 1 97644 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97643 .coefficient)
      LeftBound97641.bound (LeftBound97641.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events381.exact97642RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97641.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97641.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97644 .coefficient)
      LeftBound97622.bound (LeftBound97622.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events381.exact97627RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97622.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97622.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound97641.bound, LeftBound97622.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97641.bound, LeftBound97622.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound97641.actual selector witness, LeftBound97622.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound97645

namespace LeftBound97658
def owner : Owner := ⟨.program ⟨214⟩, ⟨25131⟩⟩
def transferEvent : Nat := 97658
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 97656 .coefficient, .predecessor 1 97657 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97656 .coefficient)
      LeftBound97503.bound (LeftBound97503.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events381.exact97655RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97503.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97503.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97657 .coefficient)
      LeftBound97486.bound (LeftBound97486.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events380.exact97493RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97486.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97486.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound97503.bound, LeftBound97486.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97503.bound, LeftBound97486.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound97503.actual selector witness, LeftBound97486.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound97658

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
