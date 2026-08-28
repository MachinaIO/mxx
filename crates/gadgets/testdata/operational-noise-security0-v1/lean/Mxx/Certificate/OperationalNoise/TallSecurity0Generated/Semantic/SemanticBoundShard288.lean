import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard083
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard084
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard234
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard237
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard287

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound42780
def owner : Owner := ⟨.program ⟨214⟩, ⟨27244⟩⟩
def transferEvent : Nat := 42780
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 42778 .coefficient, .predecessor 1 42779 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42778 .coefficient)
      LeftBound42609.bound (LeftBound42609.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events167.exact42777RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42609.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42609.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42779 .coefficient)
      LeftBound42592.bound (LeftBound42592.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events166.exact42599RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42592.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42592.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound42609.bound, LeftBound42592.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42609.bound, LeftBound42592.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound42609.actual selector witness, LeftBound42592.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound42780

namespace LeftBound42783
def owner : Owner := ⟨.program ⟨214⟩, ⟨27244⟩⟩
def transferEvent : Nat := 42783
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 42777 .summary, .result 42599 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 42777 .summary)
      LeftBound42611.bound (LeftBound42611.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20979⟩⟩) (rawTerms := some (Proof.Events167.exact42777RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound42611.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 42599 .summary)
      LeftBound42594.bound (LeftBound42594.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27243⟩⟩) (rawTerms := some (Proof.Events166.exact42599RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound42594.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound42611.bound, LeftBound42594.bound]
def bound : CoeffClass := .finite ⟨1291978824159503986688, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42611.bound, LeftBound42594.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound42611.actual selector witness, LeftBound42594.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound42783

namespace LeftBound42807
def owner : Owner := ⟨.program ⟨214⟩, ⟨11142⟩⟩
def transferEvent : Nat := 42807
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 42805 .coefficient) (.predecessor 1 42806 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42805 .coefficient)
      LeftAuthority1911.bound (LeftAuthority1911.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events007.exact1912RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1911.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1911.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42806 .coefficient)
      LeftBound36043.bound (LeftBound36043.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact36045RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36043.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36043.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority1911.bound LeftBound36043.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1911.bound, LeftBound36043.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority1911.actual selector witness) * (LeftBound36043.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound42807

namespace LeftBound42812
def owner : Owner := ⟨.program ⟨214⟩, ⟨7307⟩⟩
def transferEvent : Nat := 42812
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 42810 .coefficient) (.predecessor 1 42811 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42810 .coefficient)
      LeftBound35914.bound (LeftBound35914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact35915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35914.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35914.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42811 .coefficient)
      LeftBound13485.bound (LeftBound13485.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13486RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13485.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13485.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound35914.bound LeftBound13485.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35914.bound, LeftBound13485.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound35914.actual selector witness) * (LeftBound13485.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound42812

namespace LeftBound42817
def owner : Owner := ⟨.program ⟨214⟩, ⟨11143⟩⟩
def transferEvent : Nat := 42817
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 42815 .coefficient, .predecessor 1 42816 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42815 .coefficient)
      LeftBound42812.bound (LeftBound42812.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events167.exact42814RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42812.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42812.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42816 .coefficient)
      LeftBound42807.bound (LeftBound42807.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events167.exact42809RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42807.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42807.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound42812.bound, LeftBound42807.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42812.bound, LeftBound42807.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound42812.actual selector witness, LeftBound42807.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound42817

namespace LeftBound42821
def owner : Owner := ⟨.program ⟨214⟩, ⟨11144⟩⟩
def transferEvent : Nat := 42821
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 42819 .coefficient, .predecessor 1 42820 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42819 .coefficient)
      LeftBound42817.bound (LeftBound42817.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events167.exact42818RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42817.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42817.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42820 .coefficient)
      LeftBound13477.bound (LeftBound13477.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13478RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13477.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13477.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound42817.bound, LeftBound13477.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42817.bound, LeftBound13477.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound42817.actual selector witness, LeftBound13477.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound42821

namespace LeftBound42822
def owner : Owner := ⟨.program ⟨214⟩, ⟨11144⟩⟩
def transferEvent : Nat := 42822
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨89⟩⟩]⟩ [⟨.result 13478 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13478 .coefficient)
      LeftBound13477.bound (LeftBound13477.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨89⟩⟩) (rawTerms := some (Proof.Events052.exact13478RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13477.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13477.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound13477.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13477.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound13477.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound42822

namespace LeftBound42827
def owner : Owner := ⟨.program ⟨214⟩, ⟨12184⟩⟩
def transferEvent : Nat := 42827
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 42825 .coefficient) (.predecessor 1 42826 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42825 .coefficient)
      LeftBound42821.bound (LeftBound42821.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events167.exact42824RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42821.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42821.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42826 .coefficient)
      LeftAuthority1914.bound (LeftAuthority1914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events007.exact1915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1914.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1914.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound42821.bound LeftAuthority1914.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42821.bound, LeftAuthority1914.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound42821.actual selector witness) * (LeftAuthority1914.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound42827

namespace LeftBound42828
def owner : Owner := ⟨.program ⟨214⟩, ⟨12184⟩⟩
def transferEvent : Nat := 42828
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨12181⟩⟩], []⟩ [⟨.result 1915 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 1915 .coefficient)
      LeftAuthority1914.bound (LeftAuthority1914.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨12181⟩⟩) (rawTerms := some (Proof.Events007.exact1915RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1914.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1914.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority1914.bound []
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1914.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority1914.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound42828

namespace LeftBound42829
def owner : Owner := ⟨.program ⟨214⟩, ⟨12184⟩⟩
def transferEvent : Nat := 42829
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 42824 .summary) (.transfer 42828) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 42824 .summary)
      LeftBound42822.bound (LeftBound42822.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11144⟩⟩) (rawTerms := some (Proof.Events167.exact42824RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound42822.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 42828)
      LeftBound42828.bound (LeftBound42828.actual selector witness) := by
  exact .transfer (LeftBound42828.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound42822.bound LeftBound42828.bound
def bound : CoeffClass := .finite ⟨4992, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42822.bound, LeftBound42828.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound42822.actual selector witness) * (LeftBound42828.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound42829

namespace LeftBound42835
def owner : Owner := ⟨.program ⟨214⟩, ⟨12185⟩⟩
def transferEvent : Nat := 42835
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 42833 .coefficient) (.predecessor 1 42834 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42833 .coefficient)
      LeftAuthority1914.bound (LeftAuthority1914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events007.exact1915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1914.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1914.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42834 .coefficient)
      LeftBound36043.bound (LeftBound36043.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact36045RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36043.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36043.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority1914.bound LeftBound36043.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1914.bound, LeftBound36043.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority1914.actual selector witness) * (LeftBound36043.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound42835

namespace LeftBound42840
def owner : Owner := ⟨.program ⟨214⟩, ⟨7324⟩⟩
def transferEvent : Nat := 42840
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 42838 .coefficient) (.predecessor 1 42839 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42838 .coefficient)
      LeftBound35914.bound (LeftBound35914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact35915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35914.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35914.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42839 .coefficient)
      LeftBound13526.bound (LeftBound13526.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13527RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13526.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13526.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound35914.bound LeftBound13526.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35914.bound, LeftBound13526.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound35914.actual selector witness) * (LeftBound13526.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound42840

namespace LeftBound42845
def owner : Owner := ⟨.program ⟨214⟩, ⟨12186⟩⟩
def transferEvent : Nat := 42845
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 42843 .coefficient, .predecessor 1 42844 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42843 .coefficient)
      LeftBound42840.bound (LeftBound42840.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events167.exact42842RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42840.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42840.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42844 .coefficient)
      LeftBound42835.bound (LeftBound42835.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events167.exact42837RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42835.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42835.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound42840.bound, LeftBound42835.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42840.bound, LeftBound42835.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound42840.actual selector witness, LeftBound42835.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound42845

namespace LeftBound42849
def owner : Owner := ⟨.program ⟨214⟩, ⟨12187⟩⟩
def transferEvent : Nat := 42849
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 42847 .coefficient, .predecessor 1 42848 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42847 .coefficient)
      LeftBound42845.bound (LeftBound42845.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events167.exact42846RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42845.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42845.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42848 .coefficient)
      LeftBound13518.bound (LeftBound13518.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13519RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13518.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13518.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound42845.bound, LeftBound13518.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42845.bound, LeftBound13518.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound42845.actual selector witness, LeftBound13518.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound42849

namespace LeftBound42850
def owner : Owner := ⟨.program ⟨214⟩, ⟨12187⟩⟩
def transferEvent : Nat := 42850
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨106⟩⟩]⟩ [⟨.result 13519 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13519 .coefficient)
      LeftBound13518.bound (LeftBound13518.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨106⟩⟩) (rawTerms := some (Proof.Events052.exact13519RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13518.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13518.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound13518.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13518.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound13518.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound42850

namespace LeftBound42855
def owner : Owner := ⟨.program ⟨214⟩, ⟨12188⟩⟩
def transferEvent : Nat := 42855
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 42853 .coefficient) (.predecessor 1 42854 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42853 .coefficient)
      LeftBound42849.bound (LeftBound42849.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events167.exact42852RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42849.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42849.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42854 .coefficient)
      LeftBound13515.bound (LeftBound13515.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13516RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13515.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13515.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound42849.bound LeftBound13515.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42849.bound, LeftBound13515.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound42849.actual selector witness) * (LeftBound13515.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound42855

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
