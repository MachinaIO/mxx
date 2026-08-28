import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard653
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard714

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound104354
def owner : Owner := ⟨.program ⟨214⟩, ⟨29345⟩⟩
def transferEvent : Nat := 104354
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29343⟩⟩]⟩ [⟨.result 104350 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 104350 .coefficient)
      LeftAuthority104349.bound (LeftAuthority104349.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29343⟩⟩) (rawTerms := some (Proof.Events407.exact104350RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104349.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104349.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority104349.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority104349.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority104349.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound104354

namespace LeftBound104355
def owner : Owner := ⟨.program ⟨214⟩, ⟨29345⟩⟩
def transferEvent : Nat := 104355
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 95926 .summary) (.transfer 104354) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 95926 .summary)
      LeftBound95925.bound (LeftBound95925.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25516⟩⟩) (rawTerms := some (Proof.Events374.exact95926RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound95925.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 104354)
      LeftBound104354.bound (LeftBound104354.actual selector witness) := by
  exact .transfer (LeftBound104354.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound95925.bound LeftBound104354.bound
def bound : CoeffClass := .finite ⟨1292382246358571024384, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95925.bound, LeftBound104354.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound95925.actual selector witness) * (LeftBound104354.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104355

namespace LeftBound104366
def owner : Owner := ⟨.program ⟨214⟩, ⟨22327⟩⟩
def transferEvent : Nat := 104366
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 104364 .coefficient) (.value (.predecessor 1 104365 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104364 .coefficient)
      LeftAuthority104362.bound (LeftAuthority104362.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events407.exact104363RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104362.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104362.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104365 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority104362.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority104362.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority104362.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound104366

namespace LeftBound104370
def owner : Owner := ⟨.program ⟨214⟩, ⟨22328⟩⟩
def transferEvent : Nat := 104370
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 104368 .coefficient) (.predecessor 1 104369 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104368 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104369 .coefficient)
      LeftBound104366.bound (LeftBound104366.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events407.exact104367RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104366.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104366.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound104366.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound104366.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound104366.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104370

namespace LeftBound104371
def owner : Owner := ⟨.program ⟨214⟩, ⟨22328⟩⟩
def transferEvent : Nat := 104371
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22325⟩⟩]⟩ [⟨.result 104363 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 104363 .coefficient)
      LeftAuthority104362.bound (LeftAuthority104362.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22325⟩⟩) (rawTerms := some (Proof.Events407.exact104363RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104362.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104362.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority104362.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority104362.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority104362.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound104371

namespace LeftBound104372
def owner : Owner := ⟨.program ⟨214⟩, ⟨22328⟩⟩
def transferEvent : Nat := 104372
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 104371) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 104371)
      LeftBound104371.bound (LeftBound104371.actual selector witness) := by
  exact .transfer (LeftBound104371.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound104371.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound104371.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound104371.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104372

namespace LeftBound104443
def owner : Owner := ⟨.program ⟨214⟩, ⟨16624⟩⟩
def transferEvent : Nat := 104443
def frameStart : Nat := 104416
def rule : BoundRule := .identity (.predecessor 0 104442 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104442 .coefficient)
      LeftAuthority104440.bound (LeftAuthority104440.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events407.exact104441RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104440.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104440.derived selector witness)

def rawBound : CoeffClass := LeftAuthority104440.bound
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority104440.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority104440.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound104443

namespace LeftBound104460
def owner : Owner := ⟨.program ⟨214⟩, ⟨16700⟩⟩
def transferEvent : Nat := 104460
def frameStart : Nat := 104416
def rule : BoundRule := .sum [.predecessor 0 104458 .coefficient, .predecessor 1 104459 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104458 .coefficient)
      LeftBound104443.bound (LeftBound104443.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound104443.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104459 .coefficient)
      LeftAuthority104456.bound (LeftAuthority104456.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority104456.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104443.bound, LeftAuthority104456.bound]
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104443.bound, LeftAuthority104456.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound104443.actual selector witness, LeftAuthority104456.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104460

namespace LeftBound104463
def owner : Owner := ⟨.program ⟨214⟩, ⟨16701⟩⟩
def transferEvent : Nat := 104463
def frameStart : Nat := 104416
def rule : BoundRule := .identity (.predecessor 0 104462 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104462 .coefficient)
      LeftBound104460.bound (LeftBound104460.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound104460.derived selector witness)

def rawBound : CoeffClass := LeftBound104460.bound
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104460.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound104460.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound104463

namespace LeftBound104469
def owner : Owner := ⟨.program ⟨214⟩, ⟨16702⟩⟩
def transferEvent : Nat := 104469
def frameStart : Nat := 104416
def rule : BoundRule := .product (.predecessor 0 104467 .coefficient) (.predecessor 1 104468 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104467 .coefficient)
      LeftAuthority104465.bound (LeftAuthority104465.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events408.exact104466RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104465.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104465.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104468 .coefficient)
      LeftBound104463.bound (LeftBound104463.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events408.exact104464RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104463.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104463.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority104465.bound LeftBound104463.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority104465.bound, LeftBound104463.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority104465.actual selector witness) * (LeftBound104463.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104469

namespace LeftBound104477
def owner : Owner := ⟨.program ⟨214⟩, ⟨16703⟩⟩
def transferEvent : Nat := 104477
def frameStart : Nat := 104416
def rule : BoundRule := .sum [.predecessor 0 104475 .coefficient, .predecessor 1 104476 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104475 .coefficient)
      LeftAuthority104473.bound (LeftAuthority104473.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events408.exact104474RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104473.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104473.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104476 .coefficient)
      LeftBound104469.bound (LeftBound104469.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events408.exact104471RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104469.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104469.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority104473.bound, LeftBound104469.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority104473.bound, LeftBound104469.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority104473.actual selector witness, LeftBound104469.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104477

namespace LeftBound104481
def owner : Owner := ⟨.program ⟨214⟩, ⟨29344⟩⟩
def transferEvent : Nat := 104481
def frameStart : Nat := 104416
def rule : BoundRule := .product (.predecessor 0 104479 .coefficient) (.predecessor 1 104480 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104479 .coefficient)
      LeftBound104477.bound (LeftBound104477.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events408.exact104478RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104477.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104477.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104480 .coefficient)
      LeftAuthority104454.bound (LeftAuthority104454.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events408.exact104455RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104454.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104454.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound104477.bound LeftAuthority104454.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104477.bound, LeftAuthority104454.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound104477.actual selector witness) * (LeftAuthority104454.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104481

namespace LeftBound104492
def owner : Owner := ⟨.program ⟨214⟩, ⟨17710⟩⟩
def transferEvent : Nat := 104492
def frameStart : Nat := 104416
def rule : BoundRule := .product (.predecessor 0 104490 .coefficient) (.predecessor 1 104491 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104490 .coefficient)
      LeftAuthority104465.bound (LeftAuthority104465.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events408.exact104466RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104465.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104465.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104491 .coefficient)
      LeftAuthority104488.bound (LeftAuthority104488.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events408.exact104489RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104488.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104488.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority104465.bound LeftAuthority104488.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority104465.bound, LeftAuthority104488.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority104465.actual selector witness) * (LeftAuthority104488.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104492

namespace LeftBound104500
def owner : Owner := ⟨.program ⟨214⟩, ⟨17711⟩⟩
def transferEvent : Nat := 104500
def frameStart : Nat := 104416
def rule : BoundRule := .sum [.predecessor 0 104498 .coefficient, .predecessor 1 104499 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104498 .coefficient)
      LeftAuthority104496.bound (LeftAuthority104496.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events408.exact104497RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104496.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104496.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104499 .coefficient)
      LeftBound104492.bound (LeftBound104492.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events408.exact104494RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104492.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104492.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority104496.bound, LeftBound104492.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority104496.bound, LeftBound104492.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority104496.actual selector witness, LeftBound104492.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104500

namespace LeftBound104504
def owner : Owner := ⟨.program ⟨214⟩, ⟨29349⟩⟩
def transferEvent : Nat := 104504
def frameStart : Nat := 104416
def rule : BoundRule := .sum [.predecessor 0 104502 .coefficient, .predecessor 1 104503 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104502 .coefficient)
      LeftBound104500.bound (LeftBound104500.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events408.exact104501RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104500.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104500.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104503 .coefficient)
      LeftBound104481.bound (LeftBound104481.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events408.exact104486RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104481.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104481.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104500.bound, LeftBound104481.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104500.bound, LeftBound104481.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound104500.actual selector witness, LeftBound104481.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104504

namespace LeftBound104517
def owner : Owner := ⟨.program ⟨214⟩, ⟨29346⟩⟩
def transferEvent : Nat := 104517
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 104515 .coefficient, .predecessor 1 104516 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104515 .coefficient)
      LeftBound104370.bound (LeftBound104370.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events408.exact104514RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104370.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104370.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104516 .coefficient)
      LeftBound104353.bound (LeftBound104353.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events407.exact104360RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104353.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104353.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104370.bound, LeftBound104353.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104370.bound, LeftBound104353.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound104370.actual selector witness, LeftBound104353.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104517

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
