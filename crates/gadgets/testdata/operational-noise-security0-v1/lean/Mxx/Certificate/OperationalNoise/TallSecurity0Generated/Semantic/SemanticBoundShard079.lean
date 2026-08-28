import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard078

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound12764
def owner : Owner := ⟨.program ⟨214⟩, ⟨27486⟩⟩
def transferEvent : Nat := 12764
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 12759 .summary) (.transfer 12763) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12759 .summary)
      LeftBound12758.bound (LeftBound12758.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25934⟩⟩) (rawTerms := some (Proof.Events049.exact12759RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound12758.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 12763)
      LeftBound12763.bound (LeftBound12763.actual selector witness) := by
  exact .transfer (LeftBound12763.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound12758.bound LeftBound12763.bound
def bound : CoeffClass := .finite ⟨1292001234793221062656, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12758.bound, LeftBound12763.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound12758.actual selector witness) * (LeftBound12763.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound12764

namespace LeftBound12775
def owner : Owner := ⟨.program ⟨214⟩, ⟨21130⟩⟩
def transferEvent : Nat := 12775
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 12773 .coefficient) (.value (.predecessor 1 12774 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12773 .coefficient)
      LeftAuthority12771.bound (LeftAuthority12771.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12772RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12771.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12771.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12774 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority12771.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12771.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority12771.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound12775

namespace LeftBound12779
def owner : Owner := ⟨.program ⟨214⟩, ⟨21131⟩⟩
def transferEvent : Nat := 12779
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 12777 .coefficient) (.predecessor 1 12778 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12777 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12778 .coefficient)
      LeftBound12775.bound (LeftBound12775.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12776RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12775.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12775.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound12775.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound12775.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound12775.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound12779

namespace LeftBound12780
def owner : Owner := ⟨.program ⟨214⟩, ⟨21131⟩⟩
def transferEvent : Nat := 12780
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21128⟩⟩]⟩ [⟨.result 12772 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12772 .coefficient)
      LeftAuthority12771.bound (LeftAuthority12771.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21128⟩⟩) (rawTerms := some (Proof.Events049.exact12772RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12771.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12771.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority12771.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12771.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority12771.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound12780

namespace LeftBound12781
def owner : Owner := ⟨.program ⟨214⟩, ⟨21131⟩⟩
def transferEvent : Nat := 12781
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 12780) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 12780)
      LeftBound12780.bound (LeftBound12780.actual selector witness) := by
  exact .transfer (LeftBound12780.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound12780.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound12780.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound12780.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound12781

namespace LeftBound12876
def owner : Owner := ⟨.program ⟨214⟩, ⟨15719⟩⟩
def transferEvent : Nat := 12876
def frameStart : Nat := 12837
def rule : BoundRule := .identity (.predecessor 0 12875 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12875 .coefficient)
      LeftAuthority12873.bound (LeftAuthority12873.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact12874RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12873.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12873.derived selector witness)

def rawBound : CoeffClass := LeftAuthority12873.bound
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12873.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority12873.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound12876

namespace LeftBound12893
def owner : Owner := ⟨.program ⟨214⟩, ⟨15793⟩⟩
def transferEvent : Nat := 12893
def frameStart : Nat := 12837
def rule : BoundRule := .sum [.predecessor 0 12891 .coefficient, .predecessor 1 12892 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12891 .coefficient)
      LeftBound12876.bound (LeftBound12876.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound12876.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12892 .coefficient)
      LeftAuthority12889.bound (LeftAuthority12889.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority12889.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound12876.bound, LeftAuthority12889.bound]
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12876.bound, LeftAuthority12889.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound12876.actual selector witness, LeftAuthority12889.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound12893

namespace LeftBound12896
def owner : Owner := ⟨.program ⟨214⟩, ⟨15794⟩⟩
def transferEvent : Nat := 12896
def frameStart : Nat := 12837
def rule : BoundRule := .identity (.predecessor 0 12895 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12895 .coefficient)
      LeftBound12893.bound (LeftBound12893.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound12893.derived selector witness)

def rawBound : CoeffClass := LeftBound12893.bound
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12893.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound12893.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound12896

namespace LeftBound12902
def owner : Owner := ⟨.program ⟨214⟩, ⟨15795⟩⟩
def transferEvent : Nat := 12902
def frameStart : Nat := 12837
def rule : BoundRule := .product (.predecessor 0 12900 .coefficient) (.predecessor 1 12901 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12900 .coefficient)
      LeftAuthority12898.bound (LeftAuthority12898.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact12899RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12898.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12898.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12901 .coefficient)
      LeftBound12896.bound (LeftBound12896.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact12897RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12896.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12896.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority12898.bound LeftBound12896.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12898.bound, LeftBound12896.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority12898.actual selector witness) * (LeftBound12896.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound12902

namespace LeftBound12910
def owner : Owner := ⟨.program ⟨214⟩, ⟨15796⟩⟩
def transferEvent : Nat := 12910
def frameStart : Nat := 12837
def rule : BoundRule := .sum [.predecessor 0 12908 .coefficient, .predecessor 1 12909 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12908 .coefficient)
      LeftAuthority12906.bound (LeftAuthority12906.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact12907RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12906.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12906.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12909 .coefficient)
      LeftBound12902.bound (LeftBound12902.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact12904RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12902.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12902.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority12906.bound, LeftBound12902.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12906.bound, LeftBound12902.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority12906.actual selector witness, LeftBound12902.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound12910

namespace LeftBound12914
def owner : Owner := ⟨.program ⟨214⟩, ⟨27485⟩⟩
def transferEvent : Nat := 12914
def frameStart : Nat := 12837
def rule : BoundRule := .product (.predecessor 0 12912 .coefficient) (.predecessor 1 12913 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12912 .coefficient)
      LeftBound12910.bound (LeftBound12910.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact12911RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12910.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12910.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12913 .coefficient)
      LeftAuthority12887.bound (LeftAuthority12887.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact12888RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12887.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12887.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound12910.bound LeftAuthority12887.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12910.bound, LeftAuthority12887.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound12910.actual selector witness) * (LeftAuthority12887.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound12914

namespace LeftBound12925
def owner : Owner := ⟨.program ⟨214⟩, ⟨15761⟩⟩
def transferEvent : Nat := 12925
def frameStart : Nat := 12837
def rule : BoundRule := .product (.predecessor 0 12923 .coefficient) (.predecessor 1 12924 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12923 .coefficient)
      LeftAuthority12898.bound (LeftAuthority12898.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact12899RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12898.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12898.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12924 .coefficient)
      LeftAuthority12921.bound (LeftAuthority12921.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact12922RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12921.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12921.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority12898.bound LeftAuthority12921.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12898.bound, LeftAuthority12921.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority12898.actual selector witness) * (LeftAuthority12921.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound12925

namespace LeftBound12933
def owner : Owner := ⟨.program ⟨214⟩, ⟨15762⟩⟩
def transferEvent : Nat := 12933
def frameStart : Nat := 12837
def rule : BoundRule := .sum [.predecessor 0 12931 .coefficient, .predecessor 1 12932 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12931 .coefficient)
      LeftAuthority12929.bound (LeftAuthority12929.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact12930RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12929.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12929.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12932 .coefficient)
      LeftBound12925.bound (LeftBound12925.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact12927RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12925.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12925.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority12929.bound, LeftBound12925.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12929.bound, LeftBound12925.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority12929.actual selector witness, LeftBound12925.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound12933

namespace LeftBound12937
def owner : Owner := ⟨.program ⟨214⟩, ⟨27489⟩⟩
def transferEvent : Nat := 12937
def frameStart : Nat := 12837
def rule : BoundRule := .sum [.predecessor 0 12935 .coefficient, .predecessor 1 12936 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12935 .coefficient)
      LeftBound12933.bound (LeftBound12933.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact12934RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12933.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12933.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12936 .coefficient)
      LeftBound12914.bound (LeftBound12914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact12919RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12914.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12914.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound12933.bound, LeftBound12914.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12933.bound, LeftBound12914.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound12933.actual selector witness, LeftBound12914.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound12937

namespace LeftBound12950
def owner : Owner := ⟨.program ⟨214⟩, ⟨27487⟩⟩
def transferEvent : Nat := 12950
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 12948 .coefficient, .predecessor 1 12949 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12948 .coefficient)
      LeftBound12779.bound (LeftBound12779.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact12947RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12779.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12779.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12949 .coefficient)
      LeftBound12762.bound (LeftBound12762.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12769RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12762.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12762.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound12779.bound, LeftBound12762.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12779.bound, LeftBound12762.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound12779.actual selector witness, LeftBound12762.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound12950

namespace LeftBound12953
def owner : Owner := ⟨.program ⟨214⟩, ⟨27487⟩⟩
def transferEvent : Nat := 12953
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 12947 .summary, .result 12769 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12947 .summary)
      LeftBound12781.bound (LeftBound12781.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21131⟩⟩) (rawTerms := some (Proof.Events050.exact12947RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound12781.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12769 .summary)
      LeftBound12764.bound (LeftBound12764.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27486⟩⟩) (rawTerms := some (Proof.Events049.exact12769RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound12764.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound12781.bound, LeftBound12764.bound]
def bound : CoeffClass := .finite ⟨1292001236604524572672, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12781.bound, LeftBound12764.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound12781.actual selector witness, LeftBound12764.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound12953

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
