import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard018
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard460
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard516

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound76598
def owner : Owner := ⟨.program ⟨214⟩, ⟨16588⟩⟩
def transferEvent : Nat := 76598
def frameStart : Nat := 76525
def rule : BoundRule := .sum [.predecessor 0 76596 .coefficient, .predecessor 1 76597 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76596 .coefficient)
      LeftAuthority76594.bound (LeftAuthority76594.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events299.exact76595RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76594.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76594.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76597 .coefficient)
      LeftBound76590.bound (LeftBound76590.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events299.exact76592RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76590.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76590.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority76594.bound, LeftBound76590.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority76594.bound, LeftBound76590.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority76594.actual selector witness, LeftBound76590.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound76598

namespace LeftBound76602
def owner : Owner := ⟨.program ⟨214⟩, ⟨29149⟩⟩
def transferEvent : Nat := 76602
def frameStart : Nat := 76525
def rule : BoundRule := .product (.predecessor 0 76600 .coefficient) (.predecessor 1 76601 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76600 .coefficient)
      LeftBound76598.bound (LeftBound76598.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events299.exact76599RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76598.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76598.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76601 .coefficient)
      LeftAuthority76575.bound (LeftAuthority76575.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events299.exact76576RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76575.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76575.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound76598.bound LeftAuthority76575.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76598.bound, LeftAuthority76575.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound76598.actual selector witness) * (LeftAuthority76575.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76602

namespace LeftBound76613
def owner : Owner := ⟨.program ⟨214⟩, ⟨17947⟩⟩
def transferEvent : Nat := 76613
def frameStart : Nat := 76525
def rule : BoundRule := .product (.predecessor 0 76611 .coefficient) (.predecessor 1 76612 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76611 .coefficient)
      LeftAuthority76586.bound (LeftAuthority76586.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events299.exact76587RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76586.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76586.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76612 .coefficient)
      LeftAuthority76609.bound (LeftAuthority76609.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events299.exact76610RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76609.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76609.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority76586.bound LeftAuthority76609.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority76586.bound, LeftAuthority76609.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority76586.actual selector witness) * (LeftAuthority76609.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76613

namespace LeftBound76621
def owner : Owner := ⟨.program ⟨214⟩, ⟨17948⟩⟩
def transferEvent : Nat := 76621
def frameStart : Nat := 76525
def rule : BoundRule := .sum [.predecessor 0 76619 .coefficient, .predecessor 1 76620 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76619 .coefficient)
      LeftAuthority76617.bound (LeftAuthority76617.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events299.exact76618RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76617.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76617.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76620 .coefficient)
      LeftBound76613.bound (LeftBound76613.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events299.exact76615RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76613.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76613.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority76617.bound, LeftBound76613.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority76617.bound, LeftBound76613.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority76617.actual selector witness, LeftBound76613.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound76621

namespace LeftBound76625
def owner : Owner := ⟨.program ⟨214⟩, ⟨29154⟩⟩
def transferEvent : Nat := 76625
def frameStart : Nat := 76525
def rule : BoundRule := .sum [.predecessor 0 76623 .coefficient, .predecessor 1 76624 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76623 .coefficient)
      LeftBound76621.bound (LeftBound76621.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events299.exact76622RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76621.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76621.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76624 .coefficient)
      LeftBound76602.bound (LeftBound76602.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events299.exact76607RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76602.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76602.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound76621.bound, LeftBound76602.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76621.bound, LeftBound76602.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound76621.actual selector witness, LeftBound76602.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound76625

namespace LeftBound76638
def owner : Owner := ⟨.program ⟨214⟩, ⟨29151⟩⟩
def transferEvent : Nat := 76638
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 76636 .coefficient, .predecessor 1 76637 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76636 .coefficient)
      LeftBound76467.bound (LeftBound76467.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events299.exact76635RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76467.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76467.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76637 .coefficient)
      LeftBound76450.bound (LeftBound76450.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events298.exact76457RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76450.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76450.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound76467.bound, LeftBound76450.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76467.bound, LeftBound76450.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound76467.actual selector witness, LeftBound76450.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound76638

namespace LeftBound76641
def owner : Owner := ⟨.program ⟨214⟩, ⟨29151⟩⟩
def transferEvent : Nat := 76641
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 76635 .summary, .result 76457 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 76635 .summary)
      LeftBound76469.bound (LeftBound76469.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22191⟩⟩) (rawTerms := some (Proof.Events299.exact76635RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound76469.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 76457 .summary)
      LeftBound76452.bound (LeftBound76452.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29150⟩⟩) (rawTerms := some (Proof.Events298.exact76457RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound76452.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound76469.bound, LeftBound76452.bound]
def bound : CoeffClass := .finite ⟨1292337423279833362432, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76469.bound, LeftBound76452.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound76469.actual selector witness, LeftBound76452.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound76641

namespace LeftBound76645
def owner : Owner := ⟨.program ⟨214⟩, ⟨29152⟩⟩
def transferEvent : Nat := 76645
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 76643 .coefficient) (.predecessor 1 76644 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76643 .coefficient)
      LeftBound76638.bound (LeftBound76638.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events299.exact76642RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76638.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76638.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76644 .coefficient)
      LeftBound5598.bound (LeftBound5598.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5599RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5598.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5598.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound76638.bound LeftBound5598.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76638.bound, LeftBound5598.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound76638.actual selector witness) * (LeftBound5598.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76645

namespace LeftBound76646
def owner : Owner := ⟨.program ⟨214⟩, ⟨29152⟩⟩
def transferEvent : Nat := 76646
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6667⟩⟩]⟩ [⟨.result 5595 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5595 .coefficient)
      LeftAuthority5594.bound (LeftAuthority5594.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6667⟩⟩) (rawTerms := some (Proof.Events021.exact5595RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5594.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5594.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5594.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5594.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5594.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound76646

namespace LeftBound76647
def owner : Owner := ⟨.program ⟨214⟩, ⟨29152⟩⟩
def transferEvent : Nat := 76647
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 76642 .summary) (.transfer 76646) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 76642 .summary)
      LeftBound76641.bound (LeftBound76641.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29151⟩⟩) (rawTerms := some (Proof.Events299.exact76642RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound76641.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 76646)
      LeftBound76646.bound (LeftBound76646.actual selector witness) := by
  exact .transfer (LeftBound76646.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound76641.bound LeftBound76646.bound
def bound : CoeffClass := .finite ⟨4742899020835760917459238912, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76641.bound, LeftBound76646.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound76641.actual selector witness) * (LeftBound76646.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76647

namespace LeftBound76662
def owner : Owner := ⟨.program ⟨214⟩, ⟨28933⟩⟩
def transferEvent : Nat := 76662
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 76660 .coefficient) (.predecessor 1 76661 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76660 .coefficient)
      LeftBound67979.bound (LeftBound67979.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events265.exact67983RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67979.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67979.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76661 .coefficient)
      LeftAuthority76658.bound (LeftAuthority76658.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events299.exact76659RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76658.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76658.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound67979.bound LeftAuthority76658.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67979.bound, LeftAuthority76658.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound67979.actual selector witness) * (LeftAuthority76658.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76662

namespace LeftBound76663
def owner : Owner := ⟨.program ⟨214⟩, ⟨28933⟩⟩
def transferEvent : Nat := 76663
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28931⟩⟩]⟩ [⟨.result 76659 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 76659 .coefficient)
      LeftAuthority76658.bound (LeftAuthority76658.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28931⟩⟩) (rawTerms := some (Proof.Events299.exact76659RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76658.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76658.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority76658.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority76658.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority76658.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound76663

namespace LeftBound76664
def owner : Owner := ⟨.program ⟨214⟩, ⟨28933⟩⟩
def transferEvent : Nat := 76664
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 67983 .summary) (.transfer 76663) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 67983 .summary)
      LeftBound67982.bound (LeftBound67982.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25370⟩⟩) (rawTerms := some (Proof.Events265.exact67983RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound67982.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 76663)
      LeftBound76663.bound (LeftBound76663.actual selector witness) := by
  exact .transfer (LeftBound76663.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound67982.bound LeftBound76663.bound
def bound : CoeffClass := .finite ⟨1292315009023509266432, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67982.bound, LeftBound76663.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound67982.actual selector witness) * (LeftBound76663.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76664

namespace LeftBound76675
def owner : Owner := ⟨.program ⟨214⟩, ⟨22046⟩⟩
def transferEvent : Nat := 76675
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 76673 .coefficient) (.value (.predecessor 1 76674 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76673 .coefficient)
      LeftAuthority76671.bound (LeftAuthority76671.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events299.exact76672RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76671.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76671.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76674 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority76671.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority76671.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority76671.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound76675

namespace LeftBound76679
def owner : Owner := ⟨.program ⟨214⟩, ⟨22047⟩⟩
def transferEvent : Nat := 76679
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 76677 .coefficient) (.predecessor 1 76678 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76677 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76678 .coefficient)
      LeftBound76675.bound (LeftBound76675.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events299.exact76676RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76675.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76675.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound76675.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound76675.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound76675.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76679

namespace LeftBound76680
def owner : Owner := ⟨.program ⟨214⟩, ⟨22047⟩⟩
def transferEvent : Nat := 76680
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22044⟩⟩]⟩ [⟨.result 76672 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 76672 .coefficient)
      LeftAuthority76671.bound (LeftAuthority76671.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22044⟩⟩) (rawTerms := some (Proof.Events299.exact76672RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76671.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76671.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority76671.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority76671.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority76671.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound76680

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
