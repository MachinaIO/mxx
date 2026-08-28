import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard038

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound7700
def owner : Owner := ⟨.program ⟨214⟩, ⟨7878⟩⟩
def transferEvent : Nat := 7700
def frameStart : Nat := 7618
def rule : BoundRule := .product (.predecessor 0 7698 .coefficient) (.predecessor 1 7699 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7698 .coefficient)
      LeftBound7696.bound (LeftBound7696.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events030.exact7697RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7696.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7696.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7699 .coefficient)
      LeftBound7693.bound (LeftBound7693.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events030.exact7694RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7693.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7693.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound7696.bound LeftBound7693.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7696.bound, LeftBound7693.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound7696.actual selector witness) * (LeftBound7693.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound7700

namespace LeftBound7705
def owner : Owner := ⟨.program ⟨214⟩, ⟨13073⟩⟩
def transferEvent : Nat := 7705
def frameStart : Nat := 7618
def rule : BoundRule := .sum [.predecessor 0 7703 .coefficient, .predecessor 1 7704 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7703 .coefficient)
      LeftBound7700.bound (LeftBound7700.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events030.exact7702RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7700.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7700.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7704 .coefficient)
      LeftBound7677.bound (LeftBound7677.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7679RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7677.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7677.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound7700.bound, LeftBound7677.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7700.bound, LeftBound7677.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound7700.actual selector witness, LeftBound7677.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound7705

namespace LeftBound7709
def owner : Owner := ⟨.program ⟨214⟩, ⟨25627⟩⟩
def transferEvent : Nat := 7709
def frameStart : Nat := 7618
def rule : BoundRule := .product (.predecessor 0 7707 .coefficient) (.predecessor 1 7708 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7707 .coefficient)
      LeftBound7705.bound (LeftBound7705.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events030.exact7706RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7705.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7705.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7708 .coefficient)
      LeftAuthority7662.bound (LeftAuthority7662.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7663RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7662.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7662.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound7705.bound LeftAuthority7662.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7705.bound, LeftAuthority7662.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound7705.actual selector witness) * (LeftAuthority7662.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound7709

namespace LeftBound7720
def owner : Owner := ⟨.program ⟨214⟩, ⟨16770⟩⟩
def transferEvent : Nat := 7720
def frameStart : Nat := 7618
def rule : BoundRule := .product (.predecessor 0 7718 .coefficient) (.predecessor 1 7719 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7718 .coefficient)
      LeftAuthority7673.bound (LeftAuthority7673.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7674RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7673.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7673.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7719 .coefficient)
      LeftAuthority7716.bound (LeftAuthority7716.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events030.exact7717RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7716.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7716.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority7673.bound LeftAuthority7716.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority7673.bound, LeftAuthority7716.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority7673.actual selector witness) * (LeftAuthority7716.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound7720

namespace LeftBound7728
def owner : Owner := ⟨.program ⟨214⟩, ⟨16771⟩⟩
def transferEvent : Nat := 7728
def frameStart : Nat := 7618
def rule : BoundRule := .sum [.predecessor 0 7726 .coefficient, .predecessor 1 7727 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7726 .coefficient)
      LeftAuthority7724.bound (LeftAuthority7724.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events030.exact7725RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7724.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7724.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7727 .coefficient)
      LeftBound7720.bound (LeftBound7720.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events030.exact7722RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7720.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7720.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority7724.bound, LeftBound7720.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority7724.bound, LeftBound7720.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority7724.actual selector witness, LeftBound7720.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound7728

namespace LeftBound7732
def owner : Owner := ⟨.program ⟨214⟩, ⟨25628⟩⟩
def transferEvent : Nat := 7732
def frameStart : Nat := 7618
def rule : BoundRule := .sum [.predecessor 0 7730 .coefficient, .predecessor 1 7731 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7730 .coefficient)
      LeftBound7728.bound (LeftBound7728.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events030.exact7729RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7728.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7728.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7731 .coefficient)
      LeftBound7709.bound (LeftBound7709.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events030.exact7714RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7709.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7709.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound7728.bound, LeftBound7709.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7728.bound, LeftBound7709.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound7728.actual selector witness, LeftBound7709.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound7732

namespace LeftBound7745
def owner : Owner := ⟨.program ⟨214⟩, ⟨25626⟩⟩
def transferEvent : Nat := 7745
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 7743 .coefficient, .predecessor 1 7744 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7743 .coefficient)
      LeftBound7566.bound (LeftBound7566.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events030.exact7742RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7566.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7566.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7744 .coefficient)
      LeftBound7549.bound (LeftBound7549.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7556RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7549.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7549.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound7566.bound, LeftBound7549.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7566.bound, LeftBound7549.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound7566.actual selector witness, LeftBound7549.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound7745

namespace LeftBound7748
def owner : Owner := ⟨.program ⟨214⟩, ⟨25626⟩⟩
def transferEvent : Nat := 7748
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 7742 .summary, .result 7556 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7742 .summary)
      LeftBound7568.bound (LeftBound7568.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20123⟩⟩) (rawTerms := some (Proof.Events030.exact7742RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound7568.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7556 .summary)
      LeftBound7551.bound (LeftBound7551.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25625⟩⟩) (rawTerms := some (Proof.Events029.exact7556RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound7551.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound7568.bound, LeftBound7551.bound]
def bound : CoeffClass := .finite ⟨352164536528896, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7568.bound, LeftBound7551.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound7568.actual selector witness, LeftBound7551.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound7748

namespace LeftBound7752
def owner : Owner := ⟨.program ⟨214⟩, ⟨29656⟩⟩
def transferEvent : Nat := 7752
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 7750 .coefficient) (.predecessor 1 7751 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7750 .coefficient)
      LeftBound7745.bound (LeftBound7745.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events030.exact7749RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7745.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7745.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7751 .coefficient)
      LeftAuthority7452.bound (LeftAuthority7452.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7453RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7452.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7452.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound7745.bound LeftAuthority7452.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7745.bound, LeftAuthority7452.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound7745.actual selector witness) * (LeftAuthority7452.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound7752

namespace LeftBound7753
def owner : Owner := ⟨.program ⟨214⟩, ⟨29656⟩⟩
def transferEvent : Nat := 7753
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29654⟩⟩]⟩ [⟨.result 7453 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7453 .coefficient)
      LeftAuthority7452.bound (LeftAuthority7452.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29654⟩⟩) (rawTerms := some (Proof.Events029.exact7453RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7452.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7452.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority7452.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority7452.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority7452.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound7753

namespace LeftBound7754
def owner : Owner := ⟨.program ⟨214⟩, ⟨29656⟩⟩
def transferEvent : Nat := 7754
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 7749 .summary) (.transfer 7753) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7749 .summary)
      LeftBound7748.bound (LeftBound7748.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25626⟩⟩) (rawTerms := some (Proof.Events030.exact7749RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound7748.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 7753)
      LeftBound7753.bound (LeftBound7753.actual selector witness) := by
  exact .transfer (LeftBound7753.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound7748.bound LeftBound7753.bound
def bound : CoeffClass := .finite ⟨1292449483693632782336, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7748.bound, LeftBound7753.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound7748.actual selector witness) * (LeftBound7753.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound7754

namespace LeftBound7765
def owner : Owner := ⟨.program ⟨214⟩, ⟨22570⟩⟩
def transferEvent : Nat := 7765
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 7763 .coefficient) (.value (.predecessor 1 7764 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7763 .coefficient)
      LeftAuthority7761.bound (LeftAuthority7761.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events030.exact7762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7761.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7761.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7764 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority7761.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority7761.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority7761.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound7765

namespace LeftBound7769
def owner : Owner := ⟨.program ⟨214⟩, ⟨22571⟩⟩
def transferEvent : Nat := 7769
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 7767 .coefficient) (.predecessor 1 7768 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7767 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7768 .coefficient)
      LeftBound7765.bound (LeftBound7765.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events030.exact7766RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7765.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7765.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound7765.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound7765.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound7765.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound7769

namespace LeftBound7770
def owner : Owner := ⟨.program ⟨214⟩, ⟨22571⟩⟩
def transferEvent : Nat := 7770
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22568⟩⟩]⟩ [⟨.result 7762 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7762 .coefficient)
      LeftAuthority7761.bound (LeftAuthority7761.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22568⟩⟩) (rawTerms := some (Proof.Events030.exact7762RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7761.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7761.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority7761.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority7761.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority7761.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound7770

namespace LeftBound7771
def owner : Owner := ⟨.program ⟨214⟩, ⟨22571⟩⟩
def transferEvent : Nat := 7771
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 7770) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 7770)
      LeftBound7770.bound (LeftBound7770.actual selector witness) := by
  exact .transfer (LeftBound7770.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound7770.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound7770.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound7770.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound7771

namespace LeftBound7866
def owner : Owner := ⟨.program ⟨214⟩, ⟨16769⟩⟩
def transferEvent : Nat := 7866
def frameStart : Nat := 7827
def rule : BoundRule := .identity (.predecessor 0 7865 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7865 .coefficient)
      LeftAuthority7863.bound (LeftAuthority7863.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events030.exact7864RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7863.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7863.derived selector witness)

def rawBound : CoeffClass := LeftAuthority7863.bound
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority7863.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority7863.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound7866

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
