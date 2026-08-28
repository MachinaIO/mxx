import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound6691
def owner : Owner := ⟨.program ⟨214⟩, ⟨7883⟩⟩
def transferEvent : Nat := 6691
def frameStart : Nat := 6616
def rule : BoundRule := .scale (.predecessor 0 6689 .coefficient) (.value (.predecessor 1 6690 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6689 .coefficient)
      LeftAuthority6687.bound (LeftAuthority6687.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6688RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6687.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6687.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6690 .coefficient)
      LeftAuthority6678.bound (LeftAuthority6678.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority6678.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority6687.bound LeftAuthority6678.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6687.bound, LeftAuthority6678.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority6687.actual selector witness) * (LeftAuthority6678.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound6691

namespace LeftBound6694
def owner : Owner := ⟨.program ⟨214⟩, ⟨6770⟩⟩
def transferEvent : Nat := 6694
def frameStart : Nat := 6616
def rule : BoundRule := .identity (.predecessor 0 6693 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6693 .coefficient)
      LeftAuthority6681.bound (LeftAuthority6681.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6682RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6681.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6681.derived selector witness)

def rawBound : CoeffClass := LeftAuthority6681.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6681.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority6681.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound6694

namespace LeftBound6698
def owner : Owner := ⟨.program ⟨214⟩, ⟨7884⟩⟩
def transferEvent : Nat := 6698
def frameStart : Nat := 6616
def rule : BoundRule := .product (.predecessor 0 6696 .coefficient) (.predecessor 1 6697 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6696 .coefficient)
      LeftBound6694.bound (LeftBound6694.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6695RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6694.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6694.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6697 .coefficient)
      LeftBound6691.bound (LeftBound6691.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6692RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6691.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6691.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound6694.bound LeftBound6691.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6694.bound, LeftBound6691.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound6694.actual selector witness) * (LeftBound6691.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound6698

namespace LeftBound6703
def owner : Owner := ⟨.program ⟨214⟩, ⟨13465⟩⟩
def transferEvent : Nat := 6703
def frameStart : Nat := 6616
def rule : BoundRule := .sum [.predecessor 0 6701 .coefficient, .predecessor 1 6702 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6701 .coefficient)
      LeftBound6698.bound (LeftBound6698.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6700RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6698.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6698.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6702 .coefficient)
      LeftBound6675.bound (LeftBound6675.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6677RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6675.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6675.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6698.bound, LeftBound6675.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6698.bound, LeftBound6675.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound6698.actual selector witness, LeftBound6675.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6703

namespace LeftBound6707
def owner : Owner := ⟨.program ⟨214⟩, ⟨25781⟩⟩
def transferEvent : Nat := 6707
def frameStart : Nat := 6616
def rule : BoundRule := .product (.predecessor 0 6705 .coefficient) (.predecessor 1 6706 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6705 .coefficient)
      LeftBound6703.bound (LeftBound6703.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6704RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6703.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6703.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6706 .coefficient)
      LeftAuthority6660.bound (LeftAuthority6660.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6661RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6660.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6660.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound6703.bound LeftAuthority6660.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6703.bound, LeftAuthority6660.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound6703.actual selector witness) * (LeftAuthority6660.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound6707

namespace LeftBound6718
def owner : Owner := ⟨.program ⟨214⟩, ⟨17029⟩⟩
def transferEvent : Nat := 6718
def frameStart : Nat := 6616
def rule : BoundRule := .product (.predecessor 0 6716 .coefficient) (.predecessor 1 6717 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6716 .coefficient)
      LeftAuthority6671.bound (LeftAuthority6671.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6672RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6671.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6671.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6717 .coefficient)
      LeftAuthority6714.bound (LeftAuthority6714.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6715RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6714.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6714.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority6671.bound LeftAuthority6714.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6671.bound, LeftAuthority6714.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority6671.actual selector witness) * (LeftAuthority6714.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound6718

namespace LeftBound6726
def owner : Owner := ⟨.program ⟨214⟩, ⟨17030⟩⟩
def transferEvent : Nat := 6726
def frameStart : Nat := 6616
def rule : BoundRule := .sum [.predecessor 0 6724 .coefficient, .predecessor 1 6725 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6724 .coefficient)
      LeftAuthority6722.bound (LeftAuthority6722.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6723RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6722.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6722.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6725 .coefficient)
      LeftBound6718.bound (LeftBound6718.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6720RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6718.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6718.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority6722.bound, LeftBound6718.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6722.bound, LeftBound6718.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority6722.actual selector witness, LeftBound6718.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6726

namespace LeftBound6730
def owner : Owner := ⟨.program ⟨214⟩, ⟨25782⟩⟩
def transferEvent : Nat := 6730
def frameStart : Nat := 6616
def rule : BoundRule := .sum [.predecessor 0 6728 .coefficient, .predecessor 1 6729 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6728 .coefficient)
      LeftBound6726.bound (LeftBound6726.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6727RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6726.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6726.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6729 .coefficient)
      LeftBound6707.bound (LeftBound6707.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6712RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6707.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6707.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6726.bound, LeftBound6707.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6726.bound, LeftBound6707.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound6726.actual selector witness, LeftBound6707.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6730

namespace LeftBound6743
def owner : Owner := ⟨.program ⟨214⟩, ⟨25780⟩⟩
def transferEvent : Nat := 6743
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 6741 .coefficient, .predecessor 1 6742 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6741 .coefficient)
      LeftBound6564.bound (LeftBound6564.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6740RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6564.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6564.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6742 .coefficient)
      LeftBound6532.bound (LeftBound6532.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6539RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6532.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6532.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6564.bound, LeftBound6532.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6564.bound, LeftBound6532.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound6564.actual selector witness, LeftBound6532.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6743

namespace LeftBound6746
def owner : Owner := ⟨.program ⟨214⟩, ⟨25780⟩⟩
def transferEvent : Nat := 6746
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 6740 .summary, .result 6539 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6740 .summary)
      LeftBound6566.bound (LeftBound6566.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20267⟩⟩) (rawTerms := some (Proof.Events026.exact6740RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6566.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6539 .summary)
      LeftBound6534.bound (LeftBound6534.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25779⟩⟩) (rawTerms := some (Proof.Events025.exact6539RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6534.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6566.bound, LeftBound6534.bound]
def bound : CoeffClass := .finite ⟨352188964155392, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6566.bound, LeftBound6534.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound6566.actual selector witness, LeftBound6534.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6746

namespace LeftBound6750
def owner : Owner := ⟨.program ⟨214⟩, ⟨30207⟩⟩
def transferEvent : Nat := 6750
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 6748 .coefficient) (.predecessor 1 6749 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6748 .coefficient)
      LeftBound6743.bound (LeftBound6743.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6747RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6743.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6743.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6749 .coefficient)
      LeftAuthority6428.bound (LeftAuthority6428.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6429RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6428.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6428.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound6743.bound LeftAuthority6428.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6743.bound, LeftAuthority6428.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound6743.actual selector witness) * (LeftAuthority6428.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound6750

namespace LeftBound6751
def owner : Owner := ⟨.program ⟨214⟩, ⟨30207⟩⟩
def transferEvent : Nat := 6751
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨30205⟩⟩]⟩ [⟨.result 6429 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6429 .coefficient)
      LeftAuthority6428.bound (LeftAuthority6428.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨30205⟩⟩) (rawTerms := some (Proof.Events025.exact6429RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6428.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6428.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority6428.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6428.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority6428.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound6751

namespace LeftBound6752
def owner : Owner := ⟨.program ⟨214⟩, ⟨30207⟩⟩
def transferEvent : Nat := 6752
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6747 .summary) (.transfer 6751) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6747 .summary)
      LeftBound6746.bound (LeftBound6746.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25780⟩⟩) (rawTerms := some (Proof.Events026.exact6747RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6746.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 6751)
      LeftBound6751.bound (LeftBound6751.actual selector witness) := by
  exact .transfer (LeftBound6751.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound6746.bound LeftBound6751.bound
def bound : CoeffClass := .finite ⟨1292539133473715126272, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6746.bound, LeftBound6751.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound6746.actual selector witness) * (LeftBound6751.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound6752

namespace LeftBound6763
def owner : Owner := ⟨.program ⟨214⟩, ⟨22858⟩⟩
def transferEvent : Nat := 6763
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 6761 .coefficient) (.value (.predecessor 1 6762 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6761 .coefficient)
      LeftAuthority6759.bound (LeftAuthority6759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6760RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6759.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6762 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority6759.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6759.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority6759.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound6763

namespace LeftBound6767
def owner : Owner := ⟨.program ⟨214⟩, ⟨22859⟩⟩
def transferEvent : Nat := 6767
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 6765 .coefficient) (.predecessor 1 6766 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6765 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6766 .coefficient)
      LeftBound6763.bound (LeftBound6763.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6764RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6763.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6763.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound6763.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound6763.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound6763.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound6767

namespace LeftBound6768
def owner : Owner := ⟨.program ⟨214⟩, ⟨22859⟩⟩
def transferEvent : Nat := 6768
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22856⟩⟩]⟩ [⟨.result 6760 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6760 .coefficient)
      LeftAuthority6759.bound (LeftAuthority6759.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22856⟩⟩) (rawTerms := some (Proof.Events026.exact6760RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6759.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6759.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority6759.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6759.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority6759.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound6768

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
