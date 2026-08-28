import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound24403
def owner : Owner := ⟨.program ⟨214⟩, ⟨19830⟩⟩
def transferEvent : Nat := 24403
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 24401 .coefficient) (.value (.predecessor 1 24402 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24401 .coefficient)
      LeftAuthority24399.bound (LeftAuthority24399.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24400RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24399.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24399.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24402 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority24399.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24399.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority24399.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound24403

namespace LeftBound24407
def owner : Owner := ⟨.program ⟨214⟩, ⟨19831⟩⟩
def transferEvent : Nat := 24407
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 24405 .coefficient) (.predecessor 1 24406 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24405 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24406 .coefficient)
      LeftBound24403.bound (LeftBound24403.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24404RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24403.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24403.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound24403.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound24403.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound24403.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24407

namespace LeftBound24408
def owner : Owner := ⟨.program ⟨214⟩, ⟨19831⟩⟩
def transferEvent : Nat := 24408
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19828⟩⟩]⟩ [⟨.result 24400 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 24400 .coefficient)
      LeftAuthority24399.bound (LeftAuthority24399.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19828⟩⟩) (rawTerms := some (Proof.Events095.exact24400RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24399.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24399.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority24399.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24399.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority24399.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound24408

namespace LeftBound24409
def owner : Owner := ⟨.program ⟨214⟩, ⟨19831⟩⟩
def transferEvent : Nat := 24409
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 24408) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 24408)
      LeftBound24408.bound (LeftBound24408.actual selector witness) := by
  exact .transfer (LeftBound24408.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound24408.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound24408.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound24408.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24409

namespace LeftBound24488
def owner : Owner := ⟨.program ⟨214⟩, ⟨11982⟩⟩
def transferEvent : Nat := 24488
def frameStart : Nat := 24459
def rule : BoundRule := .product (.predecessor 0 24486 .coefficient) (.predecessor 1 24487 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24486 .coefficient)
      LeftAuthority24484.bound (LeftAuthority24484.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24485RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24484.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24484.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24487 .coefficient)
      LeftAuthority24481.bound (LeftAuthority24481.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24482RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24481.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24481.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority24484.bound LeftAuthority24481.bound
def bound : CoeffClass := .finite ⟨1296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24484.bound, LeftAuthority24481.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority24484.actual selector witness) * (LeftAuthority24481.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24488

namespace LeftBound24492
def owner : Owner := ⟨.program ⟨214⟩, ⟨11983⟩⟩
def transferEvent : Nat := 24492
def frameStart : Nat := 24459
def rule : BoundRule := .identity (.predecessor 0 24491 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24491 .coefficient)
      LeftBound24488.bound (LeftBound24488.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24490RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24488.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24488.derived selector witness)

def rawBound : CoeffClass := LeftBound24488.bound
def bound : CoeffClass := .finite ⟨1296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24488.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound24488.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound24492

namespace LeftBound24509
def owner : Owner := ⟨.program ⟨214⟩, ⟨12065⟩⟩
def transferEvent : Nat := 24509
def frameStart : Nat := 24459
def rule : BoundRule := .sum [.predecessor 0 24507 .coefficient, .predecessor 1 24508 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24507 .coefficient)
      LeftBound24492.bound (LeftBound24492.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound24492.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24508 .coefficient)
      LeftAuthority24505.bound (LeftAuthority24505.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority24505.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound24492.bound, LeftAuthority24505.bound]
def bound : CoeffClass := .finite ⟨1296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24492.bound, LeftAuthority24505.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound24492.actual selector witness, LeftAuthority24505.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24509

namespace LeftBound24512
def owner : Owner := ⟨.program ⟨214⟩, ⟨12066⟩⟩
def transferEvent : Nat := 24512
def frameStart : Nat := 24459
def rule : BoundRule := .identity (.predecessor 0 24511 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24511 .coefficient)
      LeftBound24509.bound (LeftBound24509.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound24509.derived selector witness)

def rawBound : CoeffClass := LeftBound24509.bound
def bound : CoeffClass := .finite ⟨1296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24509.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound24509.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound24512

namespace LeftBound24518
def owner : Owner := ⟨.program ⟨214⟩, ⟨12067⟩⟩
def transferEvent : Nat := 24518
def frameStart : Nat := 24459
def rule : BoundRule := .product (.predecessor 0 24516 .coefficient) (.predecessor 1 24517 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24516 .coefficient)
      LeftAuthority24514.bound (LeftAuthority24514.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24515RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24514.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24514.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24517 .coefficient)
      LeftBound24512.bound (LeftBound24512.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24513RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24512.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24512.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority24514.bound LeftBound24512.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24514.bound, LeftBound24512.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority24514.actual selector witness) * (LeftBound24512.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24518

namespace LeftBound24534
def owner : Owner := ⟨.program ⟨214⟩, ⟨7865⟩⟩
def transferEvent : Nat := 24534
def frameStart : Nat := 24459
def rule : BoundRule := .scale (.predecessor 0 24532 .coefficient) (.value (.predecessor 1 24533 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24532 .coefficient)
      LeftAuthority24530.bound (LeftAuthority24530.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24531RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24530.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24530.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24533 .coefficient)
      LeftAuthority24521.bound (LeftAuthority24521.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority24521.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority24530.bound LeftAuthority24521.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24530.bound, LeftAuthority24521.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority24530.actual selector witness) * (LeftAuthority24521.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound24534

namespace LeftBound24537
def owner : Owner := ⟨.program ⟨214⟩, ⟨6764⟩⟩
def transferEvent : Nat := 24537
def frameStart : Nat := 24459
def rule : BoundRule := .identity (.predecessor 0 24536 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24536 .coefficient)
      LeftAuthority24524.bound (LeftAuthority24524.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24525RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24524.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24524.derived selector witness)

def rawBound : CoeffClass := LeftAuthority24524.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24524.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority24524.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound24537

namespace LeftBound24541
def owner : Owner := ⟨.program ⟨214⟩, ⟨7866⟩⟩
def transferEvent : Nat := 24541
def frameStart : Nat := 24459
def rule : BoundRule := .product (.predecessor 0 24539 .coefficient) (.predecessor 1 24540 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24539 .coefficient)
      LeftBound24537.bound (LeftBound24537.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24538RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24537.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24537.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24540 .coefficient)
      LeftBound24534.bound (LeftBound24534.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24535RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24534.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24534.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound24537.bound LeftBound24534.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24537.bound, LeftBound24534.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound24537.actual selector witness) * (LeftBound24534.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24541

namespace LeftBound24546
def owner : Owner := ⟨.program ⟨214⟩, ⟨12068⟩⟩
def transferEvent : Nat := 24546
def frameStart : Nat := 24459
def rule : BoundRule := .sum [.predecessor 0 24544 .coefficient, .predecessor 1 24545 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24544 .coefficient)
      LeftBound24541.bound (LeftBound24541.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24543RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24541.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24541.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24545 .coefficient)
      LeftBound24518.bound (LeftBound24518.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24520RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24518.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24518.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound24541.bound, LeftBound24518.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24541.bound, LeftBound24518.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound24541.actual selector witness, LeftBound24518.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24546

namespace LeftBound24550
def owner : Owner := ⟨.program ⟨214⟩, ⟨25237⟩⟩
def transferEvent : Nat := 24550
def frameStart : Nat := 24459
def rule : BoundRule := .product (.predecessor 0 24548 .coefficient) (.predecessor 1 24549 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24548 .coefficient)
      LeftBound24546.bound (LeftBound24546.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24547RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24546.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24546.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24549 .coefficient)
      LeftAuthority24503.bound (LeftAuthority24503.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24504RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24503.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24503.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound24546.bound LeftAuthority24503.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24546.bound, LeftAuthority24503.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound24546.actual selector witness) * (LeftAuthority24503.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24550

namespace LeftBound24561
def owner : Owner := ⟨.program ⟨214⟩, ⟨16395⟩⟩
def transferEvent : Nat := 24561
def frameStart : Nat := 24459
def rule : BoundRule := .product (.predecessor 0 24559 .coefficient) (.predecessor 1 24560 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24559 .coefficient)
      LeftAuthority24514.bound (LeftAuthority24514.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24515RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24514.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24514.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24560 .coefficient)
      LeftAuthority24557.bound (LeftAuthority24557.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24558RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24557.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24557.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority24514.bound LeftAuthority24557.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24514.bound, LeftAuthority24557.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority24514.actual selector witness) * (LeftAuthority24557.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24561

namespace LeftBound24569
def owner : Owner := ⟨.program ⟨214⟩, ⟨16396⟩⟩
def transferEvent : Nat := 24569
def frameStart : Nat := 24459
def rule : BoundRule := .sum [.predecessor 0 24567 .coefficient, .predecessor 1 24568 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24567 .coefficient)
      LeftAuthority24565.bound (LeftAuthority24565.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24566RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24565.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24565.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24568 .coefficient)
      LeftBound24561.bound (LeftBound24561.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24563RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24561.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24561.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority24565.bound, LeftBound24561.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24565.bound, LeftBound24561.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority24565.actual selector witness, LeftBound24561.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24569

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
