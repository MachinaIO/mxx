import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard061
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard062

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound10751
def owner : Owner := ⟨.program ⟨214⟩, ⟨26242⟩⟩
def transferEvent : Nat := 10751
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 10749 .coefficient, .predecessor 1 10750 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10749 .coefficient)
      LeftBound10572.bound (LeftBound10572.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10748RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10572.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10572.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10750 .coefficient)
      LeftBound10555.bound (LeftBound10555.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10562RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10555.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10555.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound10572.bound, LeftBound10555.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10572.bound, LeftBound10555.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound10572.actual selector witness, LeftBound10555.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound10751

namespace LeftBound10754
def owner : Owner := ⟨.program ⟨214⟩, ⟨26242⟩⟩
def transferEvent : Nat := 10754
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 10748 .summary, .result 10562 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10748 .summary)
      LeftBound10574.bound (LeftBound10574.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19691⟩⟩) (rawTerms := some (Proof.Events041.exact10748RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound10574.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10562 .summary)
      LeftBound10557.bound (LeftBound10557.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26241⟩⟩) (rawTerms := some (Proof.Events041.exact10562RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound10557.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound10574.bound, LeftBound10557.bound]
def bound : CoeffClass := .finite ⟨352091253649408, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10574.bound, LeftBound10557.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound10574.actual selector witness, LeftBound10557.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound10754

namespace LeftBound10758
def owner : Owner := ⟨.program ⟨214⟩, ⟨28354⟩⟩
def transferEvent : Nat := 10758
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 10756 .coefficient) (.predecessor 1 10757 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10756 .coefficient)
      LeftBound10751.bound (LeftBound10751.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events042.exact10755RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10751.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10751.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10757 .coefficient)
      LeftAuthority10458.bound (LeftAuthority10458.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10459RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10458.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10458.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound10751.bound LeftAuthority10458.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10751.bound, LeftAuthority10458.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound10751.actual selector witness) * (LeftAuthority10458.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound10758

namespace LeftBound10759
def owner : Owner := ⟨.program ⟨214⟩, ⟨28354⟩⟩
def transferEvent : Nat := 10759
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28352⟩⟩]⟩ [⟨.result 10459 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10459 .coefficient)
      LeftAuthority10458.bound (LeftAuthority10458.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28352⟩⟩) (rawTerms := some (Proof.Events040.exact10459RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10458.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10458.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority10458.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10458.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority10458.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound10759

namespace LeftBound10760
def owner : Owner := ⟨.program ⟨214⟩, ⟨28354⟩⟩
def transferEvent : Nat := 10760
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 10755 .summary) (.transfer 10759) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10755 .summary)
      LeftBound10754.bound (LeftBound10754.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26242⟩⟩) (rawTerms := some (Proof.Events042.exact10755RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound10754.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 10759)
      LeftBound10759.bound (LeftBound10759.actual selector witness) := by
  exact .transfer (LeftBound10759.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound10754.bound LeftBound10759.bound
def bound : CoeffClass := .finite ⟨1292180534353385750528, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10754.bound, LeftBound10759.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound10754.actual selector witness) * (LeftBound10759.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound10760

namespace LeftBound10771
def owner : Owner := ⟨.program ⟨214⟩, ⟨21706⟩⟩
def transferEvent : Nat := 10771
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 10769 .coefficient) (.value (.predecessor 1 10770 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10769 .coefficient)
      LeftAuthority10767.bound (LeftAuthority10767.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events042.exact10768RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10767.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10767.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10770 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority10767.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10767.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority10767.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound10771

namespace LeftBound10775
def owner : Owner := ⟨.program ⟨214⟩, ⟨21707⟩⟩
def transferEvent : Nat := 10775
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 10773 .coefficient) (.predecessor 1 10774 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10773 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10774 .coefficient)
      LeftBound10771.bound (LeftBound10771.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events042.exact10772RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10771.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10771.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound10771.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound10771.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound10771.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound10775

namespace LeftBound10776
def owner : Owner := ⟨.program ⟨214⟩, ⟨21707⟩⟩
def transferEvent : Nat := 10776
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21704⟩⟩]⟩ [⟨.result 10768 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10768 .coefficient)
      LeftAuthority10767.bound (LeftAuthority10767.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21704⟩⟩) (rawTerms := some (Proof.Events042.exact10768RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10767.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10767.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority10767.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10767.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority10767.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound10776

namespace LeftBound10777
def owner : Owner := ⟨.program ⟨214⟩, ⟨21707⟩⟩
def transferEvent : Nat := 10777
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 10776) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 10776)
      LeftBound10776.bound (LeftBound10776.actual selector witness) := by
  exact .transfer (LeftBound10776.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound10776.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound10776.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound10776.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound10777

namespace LeftBound10872
def owner : Owner := ⟨.program ⟨214⟩, ⟨16195⟩⟩
def transferEvent : Nat := 10872
def frameStart : Nat := 10833
def rule : BoundRule := .identity (.predecessor 0 10871 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10871 .coefficient)
      LeftAuthority10869.bound (LeftAuthority10869.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events042.exact10870RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10869.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10869.derived selector witness)

def rawBound : CoeffClass := LeftAuthority10869.bound
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10869.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority10869.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound10872

namespace LeftBound10889
def owner : Owner := ⟨.program ⟨214⟩, ⟨16234⟩⟩
def transferEvent : Nat := 10889
def frameStart : Nat := 10833
def rule : BoundRule := .sum [.predecessor 0 10887 .coefficient, .predecessor 1 10888 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10887 .coefficient)
      LeftBound10872.bound (LeftBound10872.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound10872.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10888 .coefficient)
      LeftAuthority10885.bound (LeftAuthority10885.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority10885.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound10872.bound, LeftAuthority10885.bound]
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10872.bound, LeftAuthority10885.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound10872.actual selector witness, LeftAuthority10885.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound10889

namespace LeftBound10892
def owner : Owner := ⟨.program ⟨214⟩, ⟨16235⟩⟩
def transferEvent : Nat := 10892
def frameStart : Nat := 10833
def rule : BoundRule := .identity (.predecessor 0 10891 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10891 .coefficient)
      LeftBound10889.bound (LeftBound10889.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound10889.derived selector witness)

def rawBound : CoeffClass := LeftBound10889.bound
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10889.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound10889.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound10892

namespace LeftBound10898
def owner : Owner := ⟨.program ⟨214⟩, ⟨16236⟩⟩
def transferEvent : Nat := 10898
def frameStart : Nat := 10833
def rule : BoundRule := .product (.predecessor 0 10896 .coefficient) (.predecessor 1 10897 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10896 .coefficient)
      LeftAuthority10894.bound (LeftAuthority10894.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events042.exact10895RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10894.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10894.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10897 .coefficient)
      LeftBound10892.bound (LeftBound10892.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events042.exact10893RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10892.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10892.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority10894.bound LeftBound10892.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10894.bound, LeftBound10892.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority10894.actual selector witness) * (LeftBound10892.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound10898

namespace LeftBound10906
def owner : Owner := ⟨.program ⟨214⟩, ⟨16237⟩⟩
def transferEvent : Nat := 10906
def frameStart : Nat := 10833
def rule : BoundRule := .sum [.predecessor 0 10904 .coefficient, .predecessor 1 10905 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10904 .coefficient)
      LeftAuthority10902.bound (LeftAuthority10902.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events042.exact10903RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10902.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10902.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10905 .coefficient)
      LeftBound10898.bound (LeftBound10898.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events042.exact10900RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10898.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10898.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority10902.bound, LeftBound10898.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10902.bound, LeftBound10898.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority10902.actual selector witness, LeftBound10898.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound10906

namespace LeftBound10910
def owner : Owner := ⟨.program ⟨214⟩, ⟨28353⟩⟩
def transferEvent : Nat := 10910
def frameStart : Nat := 10833
def rule : BoundRule := .product (.predecessor 0 10908 .coefficient) (.predecessor 1 10909 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10908 .coefficient)
      LeftBound10906.bound (LeftBound10906.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events042.exact10907RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10906.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10906.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10909 .coefficient)
      LeftAuthority10883.bound (LeftAuthority10883.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events042.exact10884RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10883.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10883.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound10906.bound LeftAuthority10883.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10906.bound, LeftAuthority10883.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound10906.actual selector witness) * (LeftAuthority10883.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound10910

namespace LeftBound10921
def owner : Owner := ⟨.program ⟨214⟩, ⟨18403⟩⟩
def transferEvent : Nat := 10921
def frameStart : Nat := 10833
def rule : BoundRule := .product (.predecessor 0 10919 .coefficient) (.predecessor 1 10920 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10919 .coefficient)
      LeftAuthority10894.bound (LeftAuthority10894.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events042.exact10895RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10894.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10894.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10920 .coefficient)
      LeftAuthority10917.bound (LeftAuthority10917.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events042.exact10918RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10917.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10917.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority10894.bound LeftAuthority10917.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10894.bound, LeftAuthority10917.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority10894.actual selector witness) * (LeftAuthority10917.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound10921

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
