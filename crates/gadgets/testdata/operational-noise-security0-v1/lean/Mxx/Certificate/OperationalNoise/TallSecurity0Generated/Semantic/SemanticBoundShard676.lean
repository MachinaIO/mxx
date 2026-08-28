import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard068
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard675

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound98657
def owner : Owner := ⟨.program ⟨214⟩, ⟨16129⟩⟩
def transferEvent : Nat := 98657
def frameStart : Nat := 98596
def rule : BoundRule := .sum [.predecessor 0 98655 .coefficient, .predecessor 1 98656 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98655 .coefficient)
      LeftAuthority98653.bound (LeftAuthority98653.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events385.exact98654RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98653.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98653.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98656 .coefficient)
      LeftBound98649.bound (LeftBound98649.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events385.exact98651RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98649.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98649.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority98653.bound, LeftBound98649.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98653.bound, LeftBound98649.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority98653.actual selector witness, LeftBound98649.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98657

namespace LeftBound98661
def owner : Owner := ⟨.program ⟨214⟩, ⟨28049⟩⟩
def transferEvent : Nat := 98661
def frameStart : Nat := 98596
def rule : BoundRule := .product (.predecessor 0 98659 .coefficient) (.predecessor 1 98660 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98659 .coefficient)
      LeftBound98657.bound (LeftBound98657.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events385.exact98658RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98657.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98657.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98660 .coefficient)
      LeftAuthority98634.bound (LeftAuthority98634.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events385.exact98635RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98634.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98634.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound98657.bound LeftAuthority98634.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98657.bound, LeftAuthority98634.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound98657.actual selector witness) * (LeftAuthority98634.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98661

namespace LeftBound98672
def owner : Owner := ⟨.program ⟨214⟩, ⟨16099⟩⟩
def transferEvent : Nat := 98672
def frameStart : Nat := 98596
def rule : BoundRule := .product (.predecessor 0 98670 .coefficient) (.predecessor 1 98671 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98670 .coefficient)
      LeftAuthority98645.bound (LeftAuthority98645.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events385.exact98646RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98645.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98645.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98671 .coefficient)
      LeftAuthority98668.bound (LeftAuthority98668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events385.exact98669RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98668.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98668.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority98645.bound LeftAuthority98668.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98645.bound, LeftAuthority98668.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority98645.actual selector witness) * (LeftAuthority98668.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98672

namespace LeftBound98680
def owner : Owner := ⟨.program ⟨214⟩, ⟨16100⟩⟩
def transferEvent : Nat := 98680
def frameStart : Nat := 98596
def rule : BoundRule := .sum [.predecessor 0 98678 .coefficient, .predecessor 1 98679 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98678 .coefficient)
      LeftAuthority98676.bound (LeftAuthority98676.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events385.exact98677RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98676.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98676.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98679 .coefficient)
      LeftBound98672.bound (LeftBound98672.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events385.exact98674RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98672.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98672.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority98676.bound, LeftBound98672.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98676.bound, LeftBound98672.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority98676.actual selector witness, LeftBound98672.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98680

namespace LeftBound98684
def owner : Owner := ⟨.program ⟨214⟩, ⟨28053⟩⟩
def transferEvent : Nat := 98684
def frameStart : Nat := 98596
def rule : BoundRule := .sum [.predecessor 0 98682 .coefficient, .predecessor 1 98683 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98682 .coefficient)
      LeftBound98680.bound (LeftBound98680.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events385.exact98681RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98680.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98680.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98683 .coefficient)
      LeftBound98661.bound (LeftBound98661.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events385.exact98666RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98661.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98661.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound98680.bound, LeftBound98661.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98680.bound, LeftBound98661.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound98680.actual selector witness, LeftBound98661.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98684

namespace LeftBound98697
def owner : Owner := ⟨.program ⟨214⟩, ⟨28051⟩⟩
def transferEvent : Nat := 98697
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 98695 .coefficient, .predecessor 1 98696 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98695 .coefficient)
      LeftBound98550.bound (LeftBound98550.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events385.exact98694RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98550.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98550.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98696 .coefficient)
      LeftBound98533.bound (LeftBound98533.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98533.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98533.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound98550.bound, LeftBound98533.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98550.bound, LeftBound98533.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound98550.actual selector witness, LeftBound98533.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98697

namespace LeftBound98700
def owner : Owner := ⟨.program ⟨214⟩, ⟨28051⟩⟩
def transferEvent : Nat := 98700
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 98694 .summary, .result 98540 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 98694 .summary)
      LeftBound98552.bound (LeftBound98552.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21536⟩⟩) (rawTerms := some (Proof.Events385.exact98694RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound98552.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 98540 .summary)
      LeftBound98535.bound (LeftBound98535.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28050⟩⟩) (rawTerms := some (Proof.Events384.exact98540RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound98535.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound98552.bound, LeftBound98535.bound]
def bound : CoeffClass := .finite ⟨1292113298829627502592, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98552.bound, LeftBound98535.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound98552.actual selector witness, LeftBound98535.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98700

namespace LeftBound98724
def owner : Owner := ⟨.program ⟨214⟩, ⟨11458⟩⟩
def transferEvent : Nat := 98724
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 98722 .coefficient) (.predecessor 1 98723 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98722 .coefficient)
      LeftAuthority4795.bound (LeftAuthority4795.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events018.exact4796RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4795.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4795.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98723 .coefficient)
      LeftBound30.bound (LeftBound30.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact32RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority4795.bound LeftBound30.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4795.bound, LeftBound30.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority4795.actual selector witness) * (LeftBound30.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound98724

namespace LeftBound98729
def owner : Owner := ⟨.program ⟨214⟩, ⟨7116⟩⟩
def transferEvent : Nat := 98729
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 98727 .coefficient) (.predecessor 1 98728 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98727 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98728 .coefficient)
      LeftBound11481.bound (LeftBound11481.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11482RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11481.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11481.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound26.bound LeftBound11481.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftBound11481.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound26.actual selector witness) * (LeftBound11481.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98729

namespace LeftBound98734
def owner : Owner := ⟨.program ⟨214⟩, ⟨11459⟩⟩
def transferEvent : Nat := 98734
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 98732 .coefficient, .predecessor 1 98733 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98732 .coefficient)
      LeftBound98729.bound (LeftBound98729.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events385.exact98731RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98729.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98729.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98733 .coefficient)
      LeftBound98724.bound (LeftBound98724.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events385.exact98726RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98724.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98724.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound98729.bound, LeftBound98724.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98729.bound, LeftBound98724.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound98729.actual selector witness, LeftBound98724.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98734

namespace LeftBound98738
def owner : Owner := ⟨.program ⟨214⟩, ⟨11460⟩⟩
def transferEvent : Nat := 98738
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 98736 .coefficient, .predecessor 1 98737 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98736 .coefficient)
      LeftBound98734.bound (LeftBound98734.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events385.exact98735RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98734.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98734.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98737 .coefficient)
      LeftBound11473.bound (LeftBound11473.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11474RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11473.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11473.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound98734.bound, LeftBound11473.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98734.bound, LeftBound11473.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound98734.actual selector witness, LeftBound11473.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98738

namespace LeftBound98739
def owner : Owner := ⟨.program ⟨214⟩, ⟨11460⟩⟩
def transferEvent : Nat := 98739
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨93⟩⟩]⟩ [⟨.result 11474 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11474 .coefficient)
      LeftBound11473.bound (LeftBound11473.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨93⟩⟩) (rawTerms := some (Proof.Events044.exact11474RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11473.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11473.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound11473.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11473.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound11473.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound98739

namespace LeftBound98744
def owner : Owner := ⟨.program ⟨214⟩, ⟨14183⟩⟩
def transferEvent : Nat := 98744
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 98742 .coefficient) (.predecessor 1 98743 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98742 .coefficient)
      LeftBound98738.bound (LeftBound98738.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events385.exact98741RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98738.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98738.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98743 .coefficient)
      LeftAuthority4798.bound (LeftAuthority4798.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events018.exact4799RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4798.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4798.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound98738.bound LeftAuthority4798.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98738.bound, LeftAuthority4798.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound98738.actual selector witness) * (LeftAuthority4798.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98744

namespace LeftBound98745
def owner : Owner := ⟨.program ⟨214⟩, ⟨14183⟩⟩
def transferEvent : Nat := 98745
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨14180⟩⟩], []⟩ [⟨.result 4799 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 4799 .coefficient)
      LeftAuthority4798.bound (LeftAuthority4798.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨14180⟩⟩) (rawTerms := some (Proof.Events018.exact4799RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4798.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4798.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority4798.bound []
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4798.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority4798.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound98745

namespace LeftBound98746
def owner : Owner := ⟨.program ⟨214⟩, ⟨14183⟩⟩
def transferEvent : Nat := 98746
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 98741 .summary) (.transfer 98745) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 98741 .summary)
      LeftBound98739.bound (LeftBound98739.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11460⟩⟩) (rawTerms := some (Proof.Events385.exact98741RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound98739.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 98745)
      LeftBound98745.bound (LeftBound98745.actual selector witness) := by
  exact .transfer (LeftBound98745.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound98739.bound LeftBound98745.bound
def bound : CoeffClass := .finite ⟨14976, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98739.bound, LeftBound98745.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound98739.actual selector witness) * (LeftBound98745.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98746

namespace LeftBound98752
def owner : Owner := ⟨.program ⟨214⟩, ⟨14184⟩⟩
def transferEvent : Nat := 98752
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 98750 .coefficient) (.predecessor 1 98751 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98750 .coefficient)
      LeftAuthority4798.bound (LeftAuthority4798.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events018.exact4799RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4798.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4798.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98751 .coefficient)
      LeftBound30.bound (LeftBound30.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact32RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority4798.bound LeftBound30.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4798.bound, LeftBound30.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority4798.actual selector witness) * (LeftBound30.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound98752

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
