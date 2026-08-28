import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard019
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard722

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound105494
def owner : Owner := ⟨.program ⟨214⟩, ⟨21463⟩⟩
def transferEvent : Nat := 105494
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 105492 .coefficient) (.value (.predecessor 1 105493 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105492 .coefficient)
      LeftAuthority105490.bound (LeftAuthority105490.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events412.exact105491RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105490.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105490.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105493 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority105490.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105490.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority105490.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound105494

namespace LeftBound105498
def owner : Owner := ⟨.program ⟨214⟩, ⟨21464⟩⟩
def transferEvent : Nat := 105498
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 105496 .coefficient) (.predecessor 1 105497 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105496 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105497 .coefficient)
      LeftBound105494.bound (LeftBound105494.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events412.exact105495RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105494.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105494.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound105494.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound105494.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound105494.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105498

namespace LeftBound105499
def owner : Owner := ⟨.program ⟨214⟩, ⟨21464⟩⟩
def transferEvent : Nat := 105499
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21461⟩⟩]⟩ [⟨.result 105491 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 105491 .coefficient)
      LeftAuthority105490.bound (LeftAuthority105490.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21461⟩⟩) (rawTerms := some (Proof.Events412.exact105491RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105490.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105490.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority105490.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105490.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority105490.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound105499

namespace LeftBound105500
def owner : Owner := ⟨.program ⟨214⟩, ⟨21464⟩⟩
def transferEvent : Nat := 105500
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 105499) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 105499)
      LeftBound105499.bound (LeftBound105499.actual selector witness) := by
  exact .transfer (LeftBound105499.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound105499.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound105499.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound105499.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105500

namespace LeftBound105571
def owner : Owner := ⟨.program ⟨214⟩, ⟨16050⟩⟩
def transferEvent : Nat := 105571
def frameStart : Nat := 105544
def rule : BoundRule := .identity (.predecessor 0 105570 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105570 .coefficient)
      LeftAuthority105568.bound (LeftAuthority105568.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events412.exact105569RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105568.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105568.derived selector witness)

def rawBound : CoeffClass := LeftAuthority105568.bound
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105568.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority105568.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound105571

namespace LeftBound105588
def owner : Owner := ⟨.program ⟨214⟩, ⟨16126⟩⟩
def transferEvent : Nat := 105588
def frameStart : Nat := 105544
def rule : BoundRule := .sum [.predecessor 0 105586 .coefficient, .predecessor 1 105587 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105586 .coefficient)
      LeftBound105571.bound (LeftBound105571.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound105571.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105587 .coefficient)
      LeftAuthority105584.bound (LeftAuthority105584.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority105584.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound105571.bound, LeftAuthority105584.bound]
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105571.bound, LeftAuthority105584.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound105571.actual selector witness, LeftAuthority105584.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound105588

namespace LeftBound105591
def owner : Owner := ⟨.program ⟨214⟩, ⟨16127⟩⟩
def transferEvent : Nat := 105591
def frameStart : Nat := 105544
def rule : BoundRule := .identity (.predecessor 0 105590 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105590 .coefficient)
      LeftBound105588.bound (LeftBound105588.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound105588.derived selector witness)

def rawBound : CoeffClass := LeftBound105588.bound
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105588.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound105588.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound105591

namespace LeftBound105597
def owner : Owner := ⟨.program ⟨214⟩, ⟨16128⟩⟩
def transferEvent : Nat := 105597
def frameStart : Nat := 105544
def rule : BoundRule := .product (.predecessor 0 105595 .coefficient) (.predecessor 1 105596 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105595 .coefficient)
      LeftAuthority105593.bound (LeftAuthority105593.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events412.exact105594RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105593.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105593.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105596 .coefficient)
      LeftBound105591.bound (LeftBound105591.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events412.exact105592RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105591.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105591.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority105593.bound LeftBound105591.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105593.bound, LeftBound105591.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority105593.actual selector witness) * (LeftBound105591.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105597

namespace LeftBound105605
def owner : Owner := ⟨.program ⟨214⟩, ⟨16129⟩⟩
def transferEvent : Nat := 105605
def frameStart : Nat := 105544
def rule : BoundRule := .sum [.predecessor 0 105603 .coefficient, .predecessor 1 105604 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105603 .coefficient)
      LeftAuthority105601.bound (LeftAuthority105601.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events412.exact105602RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105601.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105601.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105604 .coefficient)
      LeftBound105597.bound (LeftBound105597.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events412.exact105599RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105597.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105597.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority105601.bound, LeftBound105597.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105601.bound, LeftBound105597.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority105601.actual selector witness, LeftBound105597.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound105605

namespace LeftBound105609
def owner : Owner := ⟨.program ⟨214⟩, ⟨28042⟩⟩
def transferEvent : Nat := 105609
def frameStart : Nat := 105544
def rule : BoundRule := .product (.predecessor 0 105607 .coefficient) (.predecessor 1 105608 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105607 .coefficient)
      LeftBound105605.bound (LeftBound105605.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events412.exact105606RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105605.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105605.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105608 .coefficient)
      LeftAuthority105582.bound (LeftAuthority105582.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events412.exact105583RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105582.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105582.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound105605.bound LeftAuthority105582.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105605.bound, LeftAuthority105582.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound105605.actual selector witness) * (LeftAuthority105582.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105609

namespace LeftBound105620
def owner : Owner := ⟨.program ⟨214⟩, ⟨18021⟩⟩
def transferEvent : Nat := 105620
def frameStart : Nat := 105544
def rule : BoundRule := .product (.predecessor 0 105618 .coefficient) (.predecessor 1 105619 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105618 .coefficient)
      LeftAuthority105593.bound (LeftAuthority105593.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events412.exact105594RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105593.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105593.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105619 .coefficient)
      LeftAuthority105616.bound (LeftAuthority105616.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events412.exact105617RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105616.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105616.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority105593.bound LeftAuthority105616.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105593.bound, LeftAuthority105616.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority105593.actual selector witness) * (LeftAuthority105616.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105620

namespace LeftBound105628
def owner : Owner := ⟨.program ⟨214⟩, ⟨18022⟩⟩
def transferEvent : Nat := 105628
def frameStart : Nat := 105544
def rule : BoundRule := .sum [.predecessor 0 105626 .coefficient, .predecessor 1 105627 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105626 .coefficient)
      LeftAuthority105624.bound (LeftAuthority105624.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events412.exact105625RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105624.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105624.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105627 .coefficient)
      LeftBound105620.bound (LeftBound105620.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events412.exact105622RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105620.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105620.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority105624.bound, LeftBound105620.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105624.bound, LeftBound105620.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority105624.actual selector witness, LeftBound105620.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound105628

namespace LeftBound105632
def owner : Owner := ⟨.program ⟨214⟩, ⟨28047⟩⟩
def transferEvent : Nat := 105632
def frameStart : Nat := 105544
def rule : BoundRule := .sum [.predecessor 0 105630 .coefficient, .predecessor 1 105631 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105630 .coefficient)
      LeftBound105628.bound (LeftBound105628.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events412.exact105629RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105628.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105628.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105631 .coefficient)
      LeftBound105609.bound (LeftBound105609.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events412.exact105614RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105609.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105609.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound105628.bound, LeftBound105609.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105628.bound, LeftBound105609.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound105628.actual selector witness, LeftBound105609.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound105632

namespace LeftBound105645
def owner : Owner := ⟨.program ⟨214⟩, ⟨28044⟩⟩
def transferEvent : Nat := 105645
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 105643 .coefficient, .predecessor 1 105644 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105643 .coefficient)
      LeftBound105498.bound (LeftBound105498.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events412.exact105642RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105498.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105498.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105644 .coefficient)
      LeftBound105481.bound (LeftBound105481.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events412.exact105488RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105481.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105481.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound105498.bound, LeftBound105481.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105498.bound, LeftBound105481.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound105498.actual selector witness, LeftBound105481.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound105645

namespace LeftBound105648
def owner : Owner := ⟨.program ⟨214⟩, ⟨28044⟩⟩
def transferEvent : Nat := 105648
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 105642 .summary, .result 105488 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 105642 .summary)
      LeftBound105500.bound (LeftBound105500.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21464⟩⟩) (rawTerms := some (Proof.Events412.exact105642RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound105500.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 105488 .summary)
      LeftBound105483.bound (LeftBound105483.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28043⟩⟩) (rawTerms := some (Proof.Events412.exact105488RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound105483.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound105500.bound, LeftBound105483.bound]
def bound : CoeffClass := .finite ⟨1292113298829627502592, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105500.bound, LeftBound105483.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound105500.actual selector witness, LeftBound105483.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound105648

namespace LeftBound105652
def owner : Owner := ⟨.program ⟨214⟩, ⟨28045⟩⟩
def transferEvent : Nat := 105652
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 105650 .coefficient) (.predecessor 1 105651 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105650 .coefficient)
      LeftBound105645.bound (LeftBound105645.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events412.exact105649RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105645.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105645.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105651 .coefficient)
      LeftBound5698.bound (LeftBound5698.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5699RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5698.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5698.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound105645.bound LeftBound5698.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105645.bound, LeftBound5698.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound105645.actual selector witness) * (LeftBound5698.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105652

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
