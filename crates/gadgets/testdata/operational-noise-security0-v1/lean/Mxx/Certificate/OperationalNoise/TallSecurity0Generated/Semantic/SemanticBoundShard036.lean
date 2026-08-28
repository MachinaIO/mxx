import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard025
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard028
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard035

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound7365
def owner : Owner := ⟨.program ⟨214⟩, ⟨16888⟩⟩
def transferEvent : Nat := 7365
def frameStart : Nat := 7326
def rule : BoundRule := .identity (.predecessor 0 7364 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7364 .coefficient)
      LeftAuthority7362.bound (LeftAuthority7362.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events028.exact7363RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7362.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7362.derived selector witness)

def rawBound : CoeffClass := LeftAuthority7362.bound
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority7362.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority7362.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound7365

namespace LeftBound7382
def owner : Owner := ⟨.program ⟨214⟩, ⟨16983⟩⟩
def transferEvent : Nat := 7382
def frameStart : Nat := 7326
def rule : BoundRule := .sum [.predecessor 0 7380 .coefficient, .predecessor 1 7381 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7380 .coefficient)
      LeftBound7365.bound (LeftBound7365.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound7365.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7381 .coefficient)
      LeftAuthority7378.bound (LeftAuthority7378.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority7378.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound7365.bound, LeftAuthority7378.bound]
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7365.bound, LeftAuthority7378.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound7365.actual selector witness, LeftAuthority7378.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound7382

namespace LeftBound7385
def owner : Owner := ⟨.program ⟨214⟩, ⟨16984⟩⟩
def transferEvent : Nat := 7385
def frameStart : Nat := 7326
def rule : BoundRule := .identity (.predecessor 0 7384 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7384 .coefficient)
      LeftBound7382.bound (LeftBound7382.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound7382.derived selector witness)

def rawBound : CoeffClass := LeftBound7382.bound
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7382.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound7382.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound7385

namespace LeftBound7391
def owner : Owner := ⟨.program ⟨214⟩, ⟨16985⟩⟩
def transferEvent : Nat := 7391
def frameStart : Nat := 7326
def rule : BoundRule := .product (.predecessor 0 7389 .coefficient) (.predecessor 1 7390 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7389 .coefficient)
      LeftAuthority7387.bound (LeftAuthority7387.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events028.exact7388RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7387.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7387.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7390 .coefficient)
      LeftBound7385.bound (LeftBound7385.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events028.exact7386RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7385.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7385.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority7387.bound LeftBound7385.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority7387.bound, LeftBound7385.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority7387.actual selector witness) * (LeftBound7385.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound7391

namespace LeftBound7399
def owner : Owner := ⟨.program ⟨214⟩, ⟨16986⟩⟩
def transferEvent : Nat := 7399
def frameStart : Nat := 7326
def rule : BoundRule := .sum [.predecessor 0 7397 .coefficient, .predecessor 1 7398 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7397 .coefficient)
      LeftAuthority7395.bound (LeftAuthority7395.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events028.exact7396RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7395.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7395.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7398 .coefficient)
      LeftBound7391.bound (LeftBound7391.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events028.exact7393RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7391.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7391.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority7395.bound, LeftBound7391.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority7395.bound, LeftBound7391.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority7395.actual selector witness, LeftBound7391.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound7399

namespace LeftBound7403
def owner : Owner := ⟨.program ⟨214⟩, ⟨29872⟩⟩
def transferEvent : Nat := 7403
def frameStart : Nat := 7326
def rule : BoundRule := .product (.predecessor 0 7401 .coefficient) (.predecessor 1 7402 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7401 .coefficient)
      LeftBound7399.bound (LeftBound7399.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events028.exact7400RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7399.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7399.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7402 .coefficient)
      LeftAuthority7376.bound (LeftAuthority7376.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events028.exact7377RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7376.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7376.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound7399.bound LeftAuthority7376.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7399.bound, LeftAuthority7376.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound7399.actual selector witness) * (LeftAuthority7376.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound7403

namespace LeftBound7414
def owner : Owner := ⟨.program ⟨214⟩, ⟨17098⟩⟩
def transferEvent : Nat := 7414
def frameStart : Nat := 7326
def rule : BoundRule := .product (.predecessor 0 7412 .coefficient) (.predecessor 1 7413 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7412 .coefficient)
      LeftAuthority7387.bound (LeftAuthority7387.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events028.exact7388RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7387.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7387.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7413 .coefficient)
      LeftAuthority7410.bound (LeftAuthority7410.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events028.exact7411RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7410.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7410.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority7387.bound LeftAuthority7410.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority7387.bound, LeftAuthority7410.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority7387.actual selector witness) * (LeftAuthority7410.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound7414

namespace LeftBound7422
def owner : Owner := ⟨.program ⟨214⟩, ⟨17099⟩⟩
def transferEvent : Nat := 7422
def frameStart : Nat := 7326
def rule : BoundRule := .sum [.predecessor 0 7420 .coefficient, .predecessor 1 7421 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7420 .coefficient)
      LeftAuthority7418.bound (LeftAuthority7418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events028.exact7419RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7418.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7418.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7421 .coefficient)
      LeftBound7414.bound (LeftBound7414.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events028.exact7416RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7414.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7414.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority7418.bound, LeftBound7414.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority7418.bound, LeftBound7414.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority7418.actual selector witness, LeftBound7414.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound7422

namespace LeftBound7426
def owner : Owner := ⟨.program ⟨214⟩, ⟨29876⟩⟩
def transferEvent : Nat := 7426
def frameStart : Nat := 7326
def rule : BoundRule := .sum [.predecessor 0 7424 .coefficient, .predecessor 1 7425 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7424 .coefficient)
      LeftBound7422.bound (LeftBound7422.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events028.exact7423RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7422.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7422.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7425 .coefficient)
      LeftBound7403.bound (LeftBound7403.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events028.exact7408RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7403.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7403.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound7422.bound, LeftBound7403.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7422.bound, LeftBound7403.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound7422.actual selector witness, LeftBound7403.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound7426

namespace LeftBound7439
def owner : Owner := ⟨.program ⟨214⟩, ⟨29874⟩⟩
def transferEvent : Nat := 7439
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 7437 .coefficient, .predecessor 1 7438 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7437 .coefficient)
      LeftBound7268.bound (LeftBound7268.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7436RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7268.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7268.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7438 .coefficient)
      LeftBound7251.bound (LeftBound7251.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events028.exact7258RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7251.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7251.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound7268.bound, LeftBound7251.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7268.bound, LeftBound7251.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound7268.actual selector witness, LeftBound7251.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound7439

namespace LeftBound7442
def owner : Owner := ⟨.program ⟨214⟩, ⟨29874⟩⟩
def transferEvent : Nat := 7442
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 7436 .summary, .result 7258 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7436 .summary)
      LeftBound7270.bound (LeftBound7270.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22715⟩⟩) (rawTerms := some (Proof.Events029.exact7436RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound7270.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7258 .summary)
      LeftBound7253.bound (LeftBound7253.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29873⟩⟩) (rawTerms := some (Proof.Events028.exact7258RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound7253.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound7270.bound, LeftBound7253.bound]
def bound : CoeffClass := .finite ⟨1292516722839998050304, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7270.bound, LeftBound7253.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound7270.actual selector witness, LeftBound7253.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound7442

namespace LeftBound7465
def owner : Owner := ⟨.program ⟨214⟩, ⟨102⟩⟩
def transferEvent : Nat := 7465
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 7464 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7464 .coefficient)
      LeftAuthority6440.bound (LeftAuthority6440.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6441RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6440.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6440.derived selector witness)

def rawBound : CoeffClass := LeftAuthority6440.bound
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6440.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority6440.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound7465

namespace LeftBound7469
def owner : Owner := ⟨.program ⟨214⟩, ⟨12993⟩⟩
def transferEvent : Nat := 7469
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 7467 .coefficient) (.predecessor 1 7468 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7467 .coefficient)
      LeftAuthority96.bound (LeftAuthority96.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact97RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7468 .coefficient)
      LeftBound6447.bound (LeftBound6447.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6449RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6447.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6447.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority96.bound LeftBound6447.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96.bound, LeftBound6447.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority96.actual selector witness) * (LeftBound6447.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound7469

namespace LeftBound7473
def owner : Owner := ⟨.program ⟨214⟩, ⟨6788⟩⟩
def transferEvent : Nat := 7473
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 7472 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7472 .coefficient)
      LeftAuthority5869.bound (LeftAuthority5869.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5870RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5869.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5869.derived selector witness)

def rawBound : CoeffClass := LeftAuthority5869.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5869.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority5869.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound7473

namespace LeftBound7477
def owner : Owner := ⟨.program ⟨214⟩, ⟨7396⟩⟩
def transferEvent : Nat := 7477
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 7475 .coefficient) (.predecessor 1 7476 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7475 .coefficient)
      LeftBound6313.bound (LeftBound6313.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events024.exact6314RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6313.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6313.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7476 .coefficient)
      LeftBound7473.bound (LeftBound7473.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7474RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7473.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7473.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound6313.bound LeftBound7473.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6313.bound, LeftBound7473.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound6313.actual selector witness) * (LeftBound7473.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound7477

namespace LeftBound7482
def owner : Owner := ⟨.program ⟨214⟩, ⟨12994⟩⟩
def transferEvent : Nat := 7482
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 7480 .coefficient, .predecessor 1 7481 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7480 .coefficient)
      LeftBound7477.bound (LeftBound7477.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7479RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7477.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7477.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7481 .coefficient)
      LeftBound7469.bound (LeftBound7469.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7471RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7469.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7469.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound7477.bound, LeftBound7469.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7477.bound, LeftBound7469.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound7477.actual selector witness, LeftBound7469.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound7482

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
