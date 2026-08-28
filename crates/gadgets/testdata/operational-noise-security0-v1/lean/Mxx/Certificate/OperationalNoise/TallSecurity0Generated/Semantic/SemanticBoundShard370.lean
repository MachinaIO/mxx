import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard368
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard369

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound54775
def owner : Owner := ⟨.program ⟨214⟩, ⟨16184⟩⟩
def transferEvent : Nat := 54775
def frameStart : Nat := 54673
def rule : BoundRule := .product (.predecessor 0 54773 .coefficient) (.predecessor 1 54774 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54773 .coefficient)
      LeftAuthority54728.bound (LeftAuthority54728.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events213.exact54729RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54728.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54728.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54774 .coefficient)
      LeftAuthority54771.bound (LeftAuthority54771.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events213.exact54772RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54771.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54771.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority54728.bound LeftAuthority54771.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority54728.bound, LeftAuthority54771.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority54728.actual selector witness) * (LeftAuthority54771.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54775

namespace LeftBound54783
def owner : Owner := ⟨.program ⟨214⟩, ⟨16185⟩⟩
def transferEvent : Nat := 54783
def frameStart : Nat := 54673
def rule : BoundRule := .sum [.predecessor 0 54781 .coefficient, .predecessor 1 54782 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54781 .coefficient)
      LeftAuthority54779.bound (LeftAuthority54779.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events213.exact54780RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54779.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54779.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54782 .coefficient)
      LeftBound54775.bound (LeftBound54775.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events213.exact54777RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54775.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54775.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority54779.bound, LeftBound54775.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority54779.bound, LeftBound54775.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority54779.actual selector witness, LeftBound54775.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54783

namespace LeftBound54787
def owner : Owner := ⟨.program ⟨214⟩, ⟨26229⟩⟩
def transferEvent : Nat := 54787
def frameStart : Nat := 54673
def rule : BoundRule := .sum [.predecessor 0 54785 .coefficient, .predecessor 1 54786 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54785 .coefficient)
      LeftBound54783.bound (LeftBound54783.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events214.exact54784RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54783.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54783.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54786 .coefficient)
      LeftBound54764.bound (LeftBound54764.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events213.exact54769RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54764.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54764.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54783.bound, LeftBound54764.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54783.bound, LeftBound54764.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound54783.actual selector witness, LeftBound54764.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54787

namespace LeftBound54800
def owner : Owner := ⟨.program ⟨214⟩, ⟨26227⟩⟩
def transferEvent : Nat := 54800
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 54798 .coefficient, .predecessor 1 54799 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54798 .coefficient)
      LeftBound54621.bound (LeftBound54621.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events214.exact54797RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54621.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54621.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54799 .coefficient)
      LeftBound54604.bound (LeftBound54604.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events213.exact54611RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54604.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54604.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54621.bound, LeftBound54604.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54621.bound, LeftBound54604.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound54621.actual selector witness, LeftBound54604.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54800

namespace LeftBound54803
def owner : Owner := ⟨.program ⟨214⟩, ⟨26227⟩⟩
def transferEvent : Nat := 54803
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 54797 .summary, .result 54611 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 54797 .summary)
      LeftBound54623.bound (LeftBound54623.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19679⟩⟩) (rawTerms := some (Proof.Events214.exact54797RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound54623.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 54611 .summary)
      LeftBound54606.bound (LeftBound54606.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26226⟩⟩) (rawTerms := some (Proof.Events213.exact54611RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound54606.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54623.bound, LeftBound54606.bound]
def bound : CoeffClass := .finite ⟨352091253649408, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54623.bound, LeftBound54606.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound54623.actual selector witness, LeftBound54606.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54803

namespace LeftBound54807
def owner : Owner := ⟨.program ⟨214⟩, ⟨28315⟩⟩
def transferEvent : Nat := 54807
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 54805 .coefficient) (.predecessor 1 54806 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54805 .coefficient)
      LeftBound54800.bound (LeftBound54800.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events214.exact54804RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54800.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54800.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54806 .coefficient)
      LeftAuthority54526.bound (LeftAuthority54526.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events212.exact54527RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54526.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54526.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound54800.bound LeftAuthority54526.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54800.bound, LeftAuthority54526.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound54800.actual selector witness) * (LeftAuthority54526.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54807

namespace LeftBound54808
def owner : Owner := ⟨.program ⟨214⟩, ⟨28315⟩⟩
def transferEvent : Nat := 54808
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28313⟩⟩]⟩ [⟨.result 54527 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 54527 .coefficient)
      LeftAuthority54526.bound (LeftAuthority54526.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28313⟩⟩) (rawTerms := some (Proof.Events212.exact54527RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54526.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54526.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority54526.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority54526.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority54526.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound54808

namespace LeftBound54809
def owner : Owner := ⟨.program ⟨214⟩, ⟨28315⟩⟩
def transferEvent : Nat := 54809
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 54804 .summary) (.transfer 54808) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 54804 .summary)
      LeftBound54803.bound (LeftBound54803.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26227⟩⟩) (rawTerms := some (Proof.Events214.exact54804RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound54803.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 54808)
      LeftBound54808.bound (LeftBound54808.actual selector witness) := by
  exact .transfer (LeftBound54808.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound54803.bound LeftBound54808.bound
def bound : CoeffClass := .finite ⟨1292180534353385750528, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54803.bound, LeftBound54808.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound54803.actual selector witness) * (LeftBound54808.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54809

namespace LeftBound54820
def owner : Owner := ⟨.program ⟨214⟩, ⟨21694⟩⟩
def transferEvent : Nat := 54820
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 54818 .coefficient) (.value (.predecessor 1 54819 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54818 .coefficient)
      LeftAuthority54816.bound (LeftAuthority54816.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events214.exact54817RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54816.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54816.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54819 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority54816.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority54816.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority54816.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound54820

namespace LeftBound54824
def owner : Owner := ⟨.program ⟨214⟩, ⟨21695⟩⟩
def transferEvent : Nat := 54824
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 54822 .coefficient) (.predecessor 1 54823 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54822 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54823 .coefficient)
      LeftBound54820.bound (LeftBound54820.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events214.exact54821RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54820.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54820.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound54820.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound54820.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound54820.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54824

namespace LeftBound54825
def owner : Owner := ⟨.program ⟨214⟩, ⟨21695⟩⟩
def transferEvent : Nat := 54825
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21692⟩⟩]⟩ [⟨.result 54817 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 54817 .coefficient)
      LeftAuthority54816.bound (LeftAuthority54816.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21692⟩⟩) (rawTerms := some (Proof.Events214.exact54817RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54816.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54816.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority54816.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority54816.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority54816.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound54825

namespace LeftBound54826
def owner : Owner := ⟨.program ⟨214⟩, ⟨21695⟩⟩
def transferEvent : Nat := 54826
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 54825) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 54825)
      LeftBound54825.bound (LeftBound54825.actual selector witness) := by
  exact .transfer (LeftBound54825.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound54825.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound54825.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound54825.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54826

namespace LeftBound54921
def owner : Owner := ⟨.program ⟨214⟩, ⟨16183⟩⟩
def transferEvent : Nat := 54921
def frameStart : Nat := 54882
def rule : BoundRule := .identity (.predecessor 0 54920 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54920 .coefficient)
      LeftAuthority54918.bound (LeftAuthority54918.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events214.exact54919RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54918.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54918.derived selector witness)

def rawBound : CoeffClass := LeftAuthority54918.bound
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority54918.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority54918.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound54921

namespace LeftBound54938
def owner : Owner := ⟨.program ⟨214⟩, ⟨16222⟩⟩
def transferEvent : Nat := 54938
def frameStart : Nat := 54882
def rule : BoundRule := .sum [.predecessor 0 54936 .coefficient, .predecessor 1 54937 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54936 .coefficient)
      LeftBound54921.bound (LeftBound54921.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound54921.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54937 .coefficient)
      LeftAuthority54934.bound (LeftAuthority54934.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority54934.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54921.bound, LeftAuthority54934.bound]
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54921.bound, LeftAuthority54934.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound54921.actual selector witness, LeftAuthority54934.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54938

namespace LeftBound54941
def owner : Owner := ⟨.program ⟨214⟩, ⟨16223⟩⟩
def transferEvent : Nat := 54941
def frameStart : Nat := 54882
def rule : BoundRule := .identity (.predecessor 0 54940 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54940 .coefficient)
      LeftBound54938.bound (LeftBound54938.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound54938.derived selector witness)

def rawBound : CoeffClass := LeftBound54938.bound
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54938.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound54938.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound54941

namespace LeftBound54947
def owner : Owner := ⟨.program ⟨214⟩, ⟨16224⟩⟩
def transferEvent : Nat := 54947
def frameStart : Nat := 54882
def rule : BoundRule := .product (.predecessor 0 54945 .coefficient) (.predecessor 1 54946 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54945 .coefficient)
      LeftAuthority54943.bound (LeftAuthority54943.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events214.exact54944RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54943.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54943.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54946 .coefficient)
      LeftBound54941.bound (LeftBound54941.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events214.exact54942RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54941.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54941.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority54943.bound LeftBound54941.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority54943.bound, LeftBound54941.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority54943.actual selector witness) * (LeftBound54941.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54947

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
