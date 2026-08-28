import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard598
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard631

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound93354
def owner : Owner := ⟨.program ⟨214⟩, ⟨26995⟩⟩
def transferEvent : Nat := 93354
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 93352 .coefficient) (.predecessor 1 93353 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93352 .coefficient)
      LeftBound93347.bound (LeftBound93347.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events364.exact93351RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93347.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93347.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93353 .coefficient)
      LeftBound5798.bound (LeftBound5798.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5799RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5798.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5798.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound93347.bound LeftBound5798.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound93347.bound, LeftBound5798.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound93347.actual selector witness) * (LeftBound5798.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound93354

namespace LeftBound93355
def owner : Owner := ⟨.program ⟨214⟩, ⟨26995⟩⟩
def transferEvent : Nat := 93355
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩ [⟨.result 5795 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5795 .coefficient)
      LeftAuthority5794.bound (LeftAuthority5794.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6655⟩⟩) (rawTerms := some (Proof.Events022.exact5795RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5794.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5794.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5794.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5794.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5794.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound93355

namespace LeftBound93356
def owner : Owner := ⟨.program ⟨214⟩, ⟨26995⟩⟩
def transferEvent : Nat := 93356
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 93351 .summary) (.transfer 93355) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 93351 .summary)
      LeftBound93350.bound (LeftBound93350.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26994⟩⟩) (rawTerms := some (Proof.Events364.exact93351RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound93350.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 93355)
      LeftBound93355.bound (LeftBound93355.actual selector witness) := by
  exact .transfer (LeftBound93355.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound93350.bound LeftBound93355.bound
def bound : CoeffClass := .finite ⟨4741418448262916841427435520, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound93350.bound, LeftBound93355.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound93350.actual selector witness) * (LeftBound93355.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound93356

namespace LeftBound93371
def owner : Owner := ⟨.program ⟨214⟩, ⟨26776⟩⟩
def transferEvent : Nat := 93371
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 93369 .coefficient) (.predecessor 1 93370 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93369 .coefficient)
      LeftBound87392.bound (LeftBound87392.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87396RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87392.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87392.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93370 .coefficient)
      LeftAuthority93367.bound (LeftAuthority93367.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events364.exact93368RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93367.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93367.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound87392.bound LeftAuthority93367.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87392.bound, LeftAuthority93367.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound87392.actual selector witness) * (LeftAuthority93367.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound93371

namespace LeftBound93372
def owner : Owner := ⟨.program ⟨214⟩, ⟨26776⟩⟩
def transferEvent : Nat := 93372
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26774⟩⟩]⟩ [⟨.result 93368 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 93368 .coefficient)
      LeftAuthority93367.bound (LeftAuthority93367.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26774⟩⟩) (rawTerms := some (Proof.Events364.exact93368RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93367.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93367.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority93367.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority93367.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority93367.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound93372

namespace LeftBound93373
def owner : Owner := ⟨.program ⟨214⟩, ⟨26776⟩⟩
def transferEvent : Nat := 93373
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 87396 .summary) (.transfer 93372) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 87396 .summary)
      LeftBound87395.bound (LeftBound87395.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25067⟩⟩) (rawTerms := some (Proof.Events341.exact87396RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound87395.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 93372)
      LeftBound93372.bound (LeftBound93372.actual selector witness) := by
  exact .transfer (LeftBound93372.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound87395.bound LeftBound93372.bound
def bound : CoeffClass := .finite ⟨1291911585013138718720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87395.bound, LeftBound93372.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound87395.actual selector witness) * (LeftBound93372.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound93373

namespace LeftBound93384
def owner : Owner := ⟨.program ⟨214⟩, ⟨20610⟩⟩
def transferEvent : Nat := 93384
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 93382 .coefficient) (.value (.predecessor 1 93383 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93382 .coefficient)
      LeftAuthority93380.bound (LeftAuthority93380.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events364.exact93381RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93380.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93380.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93383 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority93380.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority93380.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority93380.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound93384

namespace LeftBound93388
def owner : Owner := ⟨.program ⟨214⟩, ⟨20611⟩⟩
def transferEvent : Nat := 93388
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 93386 .coefficient) (.predecessor 1 93387 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93386 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93387 .coefficient)
      LeftBound93384.bound (LeftBound93384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events364.exact93385RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93384.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound93384.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound93384.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound93384.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound93388

namespace LeftBound93389
def owner : Owner := ⟨.program ⟨214⟩, ⟨20611⟩⟩
def transferEvent : Nat := 93389
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20608⟩⟩]⟩ [⟨.result 93381 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 93381 .coefficient)
      LeftAuthority93380.bound (LeftAuthority93380.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20608⟩⟩) (rawTerms := some (Proof.Events364.exact93381RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93380.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93380.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority93380.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority93380.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority93380.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound93389

namespace LeftBound93390
def owner : Owner := ⟨.program ⟨214⟩, ⟨20611⟩⟩
def transferEvent : Nat := 93390
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 93389) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 93389)
      LeftBound93389.bound (LeftBound93389.actual selector witness) := by
  exact .transfer (LeftBound93389.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound93389.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound93389.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound93389.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound93390

namespace LeftBound93485
def owner : Owner := ⟨.program ⟨214⟩, ⟨15115⟩⟩
def transferEvent : Nat := 93485
def frameStart : Nat := 93446
def rule : BoundRule := .identity (.predecessor 0 93484 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93484 .coefficient)
      LeftAuthority93482.bound (LeftAuthority93482.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events365.exact93483RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93482.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93482.derived selector witness)

def rawBound : CoeffClass := LeftAuthority93482.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority93482.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority93482.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound93485

namespace LeftBound93502
def owner : Owner := ⟨.program ⟨214⟩, ⟨15154⟩⟩
def transferEvent : Nat := 93502
def frameStart : Nat := 93446
def rule : BoundRule := .sum [.predecessor 0 93500 .coefficient, .predecessor 1 93501 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93500 .coefficient)
      LeftBound93485.bound (LeftBound93485.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound93485.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93501 .coefficient)
      LeftAuthority93498.bound (LeftAuthority93498.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority93498.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound93485.bound, LeftAuthority93498.bound]
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound93485.bound, LeftAuthority93498.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound93485.actual selector witness, LeftAuthority93498.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound93502

namespace LeftBound93505
def owner : Owner := ⟨.program ⟨214⟩, ⟨15155⟩⟩
def transferEvent : Nat := 93505
def frameStart : Nat := 93446
def rule : BoundRule := .identity (.predecessor 0 93504 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93504 .coefficient)
      LeftBound93502.bound (LeftBound93502.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound93502.derived selector witness)

def rawBound : CoeffClass := LeftBound93502.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound93502.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound93502.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound93505

namespace LeftBound93511
def owner : Owner := ⟨.program ⟨214⟩, ⟨15156⟩⟩
def transferEvent : Nat := 93511
def frameStart : Nat := 93446
def rule : BoundRule := .product (.predecessor 0 93509 .coefficient) (.predecessor 1 93510 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93509 .coefficient)
      LeftAuthority93507.bound (LeftAuthority93507.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events365.exact93508RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93507.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93507.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93510 .coefficient)
      LeftBound93505.bound (LeftBound93505.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events365.exact93506RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93505.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93505.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority93507.bound LeftBound93505.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority93507.bound, LeftBound93505.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority93507.actual selector witness) * (LeftBound93505.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound93511

namespace LeftBound93519
def owner : Owner := ⟨.program ⟨214⟩, ⟨15157⟩⟩
def transferEvent : Nat := 93519
def frameStart : Nat := 93446
def rule : BoundRule := .sum [.predecessor 0 93517 .coefficient, .predecessor 1 93518 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93517 .coefficient)
      LeftAuthority93515.bound (LeftAuthority93515.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events365.exact93516RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93515.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93515.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93518 .coefficient)
      LeftBound93511.bound (LeftBound93511.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events365.exact93513RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93511.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93511.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority93515.bound, LeftBound93511.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority93515.bound, LeftBound93511.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority93515.actual selector witness, LeftBound93511.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound93519

namespace LeftBound93523
def owner : Owner := ⟨.program ⟨214⟩, ⟨26775⟩⟩
def transferEvent : Nat := 93523
def frameStart : Nat := 93446
def rule : BoundRule := .product (.predecessor 0 93521 .coefficient) (.predecessor 1 93522 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93521 .coefficient)
      LeftBound93519.bound (LeftBound93519.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events365.exact93520RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93519.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93519.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93522 .coefficient)
      LeftAuthority93496.bound (LeftAuthority93496.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events365.exact93497RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93496.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93496.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound93519.bound LeftAuthority93496.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound93519.bound, LeftAuthority93496.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound93519.actual selector witness) * (LeftAuthority93496.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound93523

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
