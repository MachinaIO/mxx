import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard181

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound27486
def owner : Owner := ⟨.program ⟨214⟩, ⟨27473⟩⟩
def transferEvent : Nat := 27486
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27471⟩⟩]⟩ [⟨.result 27205 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 27205 .coefficient)
      LeftAuthority27204.bound (LeftAuthority27204.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27471⟩⟩) (rawTerms := some (Proof.Events106.exact27205RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27204.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27204.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority27204.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27204.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority27204.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound27486

namespace LeftBound27487
def owner : Owner := ⟨.program ⟨214⟩, ⟨27473⟩⟩
def transferEvent : Nat := 27487
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 27482 .summary) (.transfer 27486) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 27482 .summary)
      LeftBound27481.bound (LeftBound27481.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25929⟩⟩) (rawTerms := some (Proof.Events107.exact27482RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound27481.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 27486)
      LeftBound27486.bound (LeftBound27486.actual selector witness) := by
  exact .transfer (LeftBound27486.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound27481.bound LeftBound27486.bound
def bound : CoeffClass := .finite ⟨1292001234793221062656, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27481.bound, LeftBound27486.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound27481.actual selector witness) * (LeftBound27486.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27487

namespace LeftBound27498
def owner : Owner := ⟨.program ⟨214⟩, ⟨21126⟩⟩
def transferEvent : Nat := 27498
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 27496 .coefficient) (.value (.predecessor 1 27497 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27496 .coefficient)
      LeftAuthority27494.bound (LeftAuthority27494.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27495RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27494.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27494.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27497 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority27494.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27494.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority27494.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound27498

namespace LeftBound27502
def owner : Owner := ⟨.program ⟨214⟩, ⟨21127⟩⟩
def transferEvent : Nat := 27502
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 27500 .coefficient) (.predecessor 1 27501 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27500 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27501 .coefficient)
      LeftBound27498.bound (LeftBound27498.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27499RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27498.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27498.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound27498.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound27498.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound27498.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27502

namespace LeftBound27503
def owner : Owner := ⟨.program ⟨214⟩, ⟨21127⟩⟩
def transferEvent : Nat := 27503
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21124⟩⟩]⟩ [⟨.result 27495 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 27495 .coefficient)
      LeftAuthority27494.bound (LeftAuthority27494.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21124⟩⟩) (rawTerms := some (Proof.Events107.exact27495RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27494.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27494.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority27494.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27494.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority27494.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound27503

namespace LeftBound27504
def owner : Owner := ⟨.program ⟨214⟩, ⟨21127⟩⟩
def transferEvent : Nat := 27504
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 27503) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 27503)
      LeftBound27503.bound (LeftBound27503.actual selector witness) := by
  exact .transfer (LeftBound27503.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound27503.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound27503.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound27503.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27504

namespace LeftBound27599
def owner : Owner := ⟨.program ⟨214⟩, ⟨15715⟩⟩
def transferEvent : Nat := 27599
def frameStart : Nat := 27560
def rule : BoundRule := .identity (.predecessor 0 27598 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27598 .coefficient)
      LeftAuthority27596.bound (LeftAuthority27596.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27597RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27596.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27596.derived selector witness)

def rawBound : CoeffClass := LeftAuthority27596.bound
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27596.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority27596.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound27599

namespace LeftBound27616
def owner : Owner := ⟨.program ⟨214⟩, ⟨15789⟩⟩
def transferEvent : Nat := 27616
def frameStart : Nat := 27560
def rule : BoundRule := .sum [.predecessor 0 27614 .coefficient, .predecessor 1 27615 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27614 .coefficient)
      LeftBound27599.bound (LeftBound27599.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound27599.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27615 .coefficient)
      LeftAuthority27612.bound (LeftAuthority27612.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority27612.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27599.bound, LeftAuthority27612.bound]
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27599.bound, LeftAuthority27612.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound27599.actual selector witness, LeftAuthority27612.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27616

namespace LeftBound27619
def owner : Owner := ⟨.program ⟨214⟩, ⟨15790⟩⟩
def transferEvent : Nat := 27619
def frameStart : Nat := 27560
def rule : BoundRule := .identity (.predecessor 0 27618 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27618 .coefficient)
      LeftBound27616.bound (LeftBound27616.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound27616.derived selector witness)

def rawBound : CoeffClass := LeftBound27616.bound
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27616.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound27616.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound27619

namespace LeftBound27625
def owner : Owner := ⟨.program ⟨214⟩, ⟨15791⟩⟩
def transferEvent : Nat := 27625
def frameStart : Nat := 27560
def rule : BoundRule := .product (.predecessor 0 27623 .coefficient) (.predecessor 1 27624 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27623 .coefficient)
      LeftAuthority27621.bound (LeftAuthority27621.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27622RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27621.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27621.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27624 .coefficient)
      LeftBound27619.bound (LeftBound27619.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27620RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27619.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27619.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority27621.bound LeftBound27619.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27621.bound, LeftBound27619.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority27621.actual selector witness) * (LeftBound27619.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27625

namespace LeftBound27633
def owner : Owner := ⟨.program ⟨214⟩, ⟨15792⟩⟩
def transferEvent : Nat := 27633
def frameStart : Nat := 27560
def rule : BoundRule := .sum [.predecessor 0 27631 .coefficient, .predecessor 1 27632 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27631 .coefficient)
      LeftAuthority27629.bound (LeftAuthority27629.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27630RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27629.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27629.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27632 .coefficient)
      LeftBound27625.bound (LeftBound27625.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27627RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27625.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27625.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority27629.bound, LeftBound27625.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27629.bound, LeftBound27625.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority27629.actual selector witness, LeftBound27625.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27633

namespace LeftBound27637
def owner : Owner := ⟨.program ⟨214⟩, ⟨27472⟩⟩
def transferEvent : Nat := 27637
def frameStart : Nat := 27560
def rule : BoundRule := .product (.predecessor 0 27635 .coefficient) (.predecessor 1 27636 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27635 .coefficient)
      LeftBound27633.bound (LeftBound27633.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27634RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27633.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27633.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27636 .coefficient)
      LeftAuthority27610.bound (LeftAuthority27610.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27611RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27610.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27610.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound27633.bound LeftAuthority27610.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27633.bound, LeftAuthority27610.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound27633.actual selector witness) * (LeftAuthority27610.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27637

namespace LeftBound27648
def owner : Owner := ⟨.program ⟨214⟩, ⟨15758⟩⟩
def transferEvent : Nat := 27648
def frameStart : Nat := 27560
def rule : BoundRule := .product (.predecessor 0 27646 .coefficient) (.predecessor 1 27647 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27646 .coefficient)
      LeftAuthority27621.bound (LeftAuthority27621.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27622RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27621.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27621.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27647 .coefficient)
      LeftAuthority27644.bound (LeftAuthority27644.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27645RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27644.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27644.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority27621.bound LeftAuthority27644.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27621.bound, LeftAuthority27644.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority27621.actual selector witness) * (LeftAuthority27644.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27648

namespace LeftBound27656
def owner : Owner := ⟨.program ⟨214⟩, ⟨15759⟩⟩
def transferEvent : Nat := 27656
def frameStart : Nat := 27560
def rule : BoundRule := .sum [.predecessor 0 27654 .coefficient, .predecessor 1 27655 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27654 .coefficient)
      LeftAuthority27652.bound (LeftAuthority27652.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events108.exact27653RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27652.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27652.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27655 .coefficient)
      LeftBound27648.bound (LeftBound27648.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events108.exact27650RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27648.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27648.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority27652.bound, LeftBound27648.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27652.bound, LeftBound27648.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority27652.actual selector witness, LeftBound27648.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27656

namespace LeftBound27660
def owner : Owner := ⟨.program ⟨214⟩, ⟨27476⟩⟩
def transferEvent : Nat := 27660
def frameStart : Nat := 27560
def rule : BoundRule := .sum [.predecessor 0 27658 .coefficient, .predecessor 1 27659 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27658 .coefficient)
      LeftBound27656.bound (LeftBound27656.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events108.exact27657RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27656.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27656.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27659 .coefficient)
      LeftBound27637.bound (LeftBound27637.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27642RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27637.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27637.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27656.bound, LeftBound27637.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27656.bound, LeftBound27637.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound27656.actual selector witness, LeftBound27637.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27660

namespace LeftBound27673
def owner : Owner := ⟨.program ⟨214⟩, ⟨27474⟩⟩
def transferEvent : Nat := 27673
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 27671 .coefficient, .predecessor 1 27672 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27671 .coefficient)
      LeftBound27502.bound (LeftBound27502.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events108.exact27670RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27502.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27502.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27672 .coefficient)
      LeftBound27485.bound (LeftBound27485.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27492RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27485.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27485.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27502.bound, LeftBound27485.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27502.bound, LeftBound27485.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound27502.actual selector witness, LeftBound27485.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27673

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
