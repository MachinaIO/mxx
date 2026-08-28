import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard072
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard132
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard135
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard175

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound26750
def owner : Owner := ⟨.program ⟨214⟩, ⟨11400⟩⟩
def transferEvent : Nat := 26750
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 26748 .coefficient, .predecessor 1 26749 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26748 .coefficient)
      LeftBound26746.bound (LeftBound26746.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events104.exact26747RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26746.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26746.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26749 .coefficient)
      LeftBound11974.bound (LeftBound11974.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11975RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11974.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11974.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26746.bound, LeftBound11974.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26746.bound, LeftBound11974.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound26746.actual selector witness, LeftBound11974.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26750

namespace LeftBound26751
def owner : Owner := ⟨.program ⟨214⟩, ⟨11400⟩⟩
def transferEvent : Nat := 26751
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨92⟩⟩]⟩ [⟨.result 11975 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11975 .coefficient)
      LeftBound11974.bound (LeftBound11974.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨92⟩⟩) (rawTerms := some (Proof.Events046.exact11975RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11974.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11974.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound11974.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11974.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound11974.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound26751

namespace LeftBound26756
def owner : Owner := ⟨.program ⟨214⟩, ⟨14020⟩⟩
def transferEvent : Nat := 26756
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 26754 .coefficient) (.predecessor 1 26755 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26754 .coefficient)
      LeftBound26750.bound (LeftBound26750.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events104.exact26753RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26750.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26750.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26755 .coefficient)
      LeftAuthority1097.bound (LeftAuthority1097.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events004.exact1098RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1097.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1097.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound26750.bound LeftAuthority1097.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26750.bound, LeftAuthority1097.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound26750.actual selector witness) * (LeftAuthority1097.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound26756

namespace LeftBound26757
def owner : Owner := ⟨.program ⟨214⟩, ⟨14020⟩⟩
def transferEvent : Nat := 26757
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨14017⟩⟩], []⟩ [⟨.result 1098 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 1098 .coefficient)
      LeftAuthority1097.bound (LeftAuthority1097.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨14017⟩⟩) (rawTerms := some (Proof.Events004.exact1098RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1097.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1097.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority1097.bound []
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1097.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority1097.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound26757

namespace LeftBound26758
def owner : Owner := ⟨.program ⟨214⟩, ⟨14020⟩⟩
def transferEvent : Nat := 26758
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 26753 .summary) (.transfer 26757) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 26753 .summary)
      LeftBound26751.bound (LeftBound26751.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11400⟩⟩) (rawTerms := some (Proof.Events104.exact26753RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound26751.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 26757)
      LeftBound26757.bound (LeftBound26757.actual selector witness) := by
  exact .transfer (LeftBound26757.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound26751.bound LeftBound26757.bound
def bound : CoeffClass := .finite ⟨13312, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26751.bound, LeftBound26757.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound26751.actual selector witness) * (LeftBound26757.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound26758

namespace LeftBound26764
def owner : Owner := ⟨.program ⟨214⟩, ⟨14021⟩⟩
def transferEvent : Nat := 26764
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 26762 .coefficient) (.predecessor 1 26763 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26762 .coefficient)
      LeftAuthority1097.bound (LeftAuthority1097.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events004.exact1098RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1097.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1097.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26763 .coefficient)
      LeftBound21418.bound (LeftBound21418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21420RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21418.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority1097.bound LeftBound21418.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1097.bound, LeftBound21418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority1097.actual selector witness) * (LeftBound21418.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound26764

namespace LeftBound26769
def owner : Owner := ⟨.program ⟨214⟩, ⟨7328⟩⟩
def transferEvent : Nat := 26769
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 26767 .coefficient) (.predecessor 1 26768 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26767 .coefficient)
      LeftBound21289.bound (LeftBound21289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21289.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21289.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26768 .coefficient)
      LeftBound12023.bound (LeftBound12023.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact12024RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12023.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12023.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound21289.bound LeftBound12023.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21289.bound, LeftBound12023.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound21289.actual selector witness) * (LeftBound12023.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound26769

namespace LeftBound26774
def owner : Owner := ⟨.program ⟨214⟩, ⟨14022⟩⟩
def transferEvent : Nat := 26774
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 26772 .coefficient, .predecessor 1 26773 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26772 .coefficient)
      LeftBound26769.bound (LeftBound26769.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events104.exact26771RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26769.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26769.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26773 .coefficient)
      LeftBound26764.bound (LeftBound26764.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events104.exact26766RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26764.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26764.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26769.bound, LeftBound26764.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26769.bound, LeftBound26764.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound26769.actual selector witness, LeftBound26764.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26774

namespace LeftBound26778
def owner : Owner := ⟨.program ⟨214⟩, ⟨14023⟩⟩
def transferEvent : Nat := 26778
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 26776 .coefficient, .predecessor 1 26777 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26776 .coefficient)
      LeftBound26774.bound (LeftBound26774.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events104.exact26775RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26774.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26774.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26777 .coefficient)
      LeftBound12015.bound (LeftBound12015.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact12016RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12015.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12015.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26774.bound, LeftBound12015.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26774.bound, LeftBound12015.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound26774.actual selector witness, LeftBound12015.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26778

namespace LeftBound26779
def owner : Owner := ⟨.program ⟨214⟩, ⟨14023⟩⟩
def transferEvent : Nat := 26779
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨72⟩⟩]⟩ [⟨.result 12016 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12016 .coefficient)
      LeftBound12015.bound (LeftBound12015.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨72⟩⟩) (rawTerms := some (Proof.Events046.exact12016RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12015.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12015.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound12015.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12015.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound12015.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound26779

namespace LeftBound26784
def owner : Owner := ⟨.program ⟨214⟩, ⟨14024⟩⟩
def transferEvent : Nat := 26784
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 26782 .coefficient) (.predecessor 1 26783 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26782 .coefficient)
      LeftBound26778.bound (LeftBound26778.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events104.exact26781RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26778.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26778.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26783 .coefficient)
      LeftBound12012.bound (LeftBound12012.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact12013RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12012.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12012.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound26778.bound LeftBound12012.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26778.bound, LeftBound12012.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound26778.actual selector witness) * (LeftBound12012.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound26784

namespace LeftBound26785
def owner : Owner := ⟨.program ⟨214⟩, ⟨14024⟩⟩
def transferEvent : Nat := 26785
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩ [⟨.result 12009 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12009 .coefficient)
      LeftAuthority12008.bound (LeftAuthority12008.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7849⟩⟩) (rawTerms := some (Proof.Events046.exact12009RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12008.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12008.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority12008.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12008.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority12008.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound26785

namespace LeftBound26786
def owner : Owner := ⟨.program ⟨214⟩, ⟨14024⟩⟩
def transferEvent : Nat := 26786
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 26781 .summary) (.transfer 26785) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 26781 .summary)
      LeftBound26779.bound (LeftBound26779.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14023⟩⟩) (rawTerms := some (Proof.Events104.exact26781RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound26779.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 26785)
      LeftBound26785.bound (LeftBound26785.actual selector witness) := by
  exact .transfer (LeftBound26785.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound26779.bound LeftBound26785.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26779.bound, LeftBound26785.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound26779.actual selector witness) * (LeftBound26785.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound26786

namespace LeftBound26794
def owner : Owner := ⟨.program ⟨214⟩, ⟨14025⟩⟩
def transferEvent : Nat := 26794
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 26792 .coefficient, .predecessor 1 26793 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26792 .coefficient)
      LeftBound26784.bound (LeftBound26784.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events104.exact26791RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26784.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26784.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26793 .coefficient)
      LeftBound26756.bound (LeftBound26756.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events104.exact26761RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26756.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26756.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26784.bound, LeftBound26756.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26784.bound, LeftBound26756.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound26784.actual selector witness, LeftBound26756.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26794

namespace LeftBound26796
def owner : Owner := ⟨.program ⟨214⟩, ⟨14025⟩⟩
def transferEvent : Nat := 26796
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 26791 .summary, .result 26761 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 26791 .summary)
      LeftBound26786.bound (LeftBound26786.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14024⟩⟩) (rawTerms := some (Proof.Events104.exact26791RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound26786.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 26761 .summary)
      LeftBound26758.bound (LeftBound26758.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14020⟩⟩) (rawTerms := some (Proof.Events104.exact26761RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound26758.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26786.bound, LeftBound26758.bound]
def bound : CoeffClass := .finite ⟨95433728, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26786.bound, LeftBound26758.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound26786.actual selector witness, LeftBound26758.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26796

namespace LeftBound26800
def owner : Owner := ⟨.program ⟨214⟩, ⟨26005⟩⟩
def transferEvent : Nat := 26800
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 26798 .coefficient) (.predecessor 1 26799 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26798 .coefficient)
      LeftBound26794.bound (LeftBound26794.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events104.exact26797RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26794.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26794.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26799 .coefficient)
      LeftAuthority26732.bound (LeftAuthority26732.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events104.exact26733RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26732.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26732.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound26794.bound LeftAuthority26732.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26794.bound, LeftAuthority26732.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound26794.actual selector witness) * (LeftAuthority26732.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound26800

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
