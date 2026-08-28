import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard076
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard234
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard237
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard280

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound41853
def owner : Owner := ⟨.program ⟨214⟩, ⟨11311⟩⟩
def transferEvent : Nat := 41853
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 41851 .coefficient, .predecessor 1 41852 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41851 .coefficient)
      LeftBound41848.bound (LeftBound41848.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41850RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41848.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41848.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41852 .coefficient)
      LeftBound41843.bound (LeftBound41843.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41845RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41843.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41843.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound41848.bound, LeftBound41843.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41848.bound, LeftBound41843.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound41848.actual selector witness, LeftBound41843.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41853

namespace LeftBound41857
def owner : Owner := ⟨.program ⟨214⟩, ⟨11312⟩⟩
def transferEvent : Nat := 41857
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 41855 .coefficient, .predecessor 1 41856 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41855 .coefficient)
      LeftBound41853.bound (LeftBound41853.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41854RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41853.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41853.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41856 .coefficient)
      LeftBound12475.bound (LeftBound12475.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events048.exact12476RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12475.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12475.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound41853.bound, LeftBound12475.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41853.bound, LeftBound12475.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound41853.actual selector witness, LeftBound12475.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41857

namespace LeftBound41858
def owner : Owner := ⟨.program ⟨214⟩, ⟨11312⟩⟩
def transferEvent : Nat := 41858
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨91⟩⟩]⟩ [⟨.result 12476 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12476 .coefficient)
      LeftBound12475.bound (LeftBound12475.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨91⟩⟩) (rawTerms := some (Proof.Events048.exact12476RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12475.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12475.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound12475.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12475.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound12475.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound41858

namespace LeftBound41863
def owner : Owner := ⟨.program ⟨214⟩, ⟨13794⟩⟩
def transferEvent : Nat := 41863
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 41861 .coefficient) (.predecessor 1 41862 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41861 .coefficient)
      LeftBound41857.bound (LeftBound41857.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41860RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41857.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41857.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41862 .coefficient)
      LeftAuthority1868.bound (LeftAuthority1868.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events007.exact1869RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1868.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1868.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound41857.bound LeftAuthority1868.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41857.bound, LeftAuthority1868.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound41857.actual selector witness) * (LeftAuthority1868.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound41863

namespace LeftBound41864
def owner : Owner := ⟨.program ⟨214⟩, ⟨13794⟩⟩
def transferEvent : Nat := 41864
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨13791⟩⟩], []⟩ [⟨.result 1869 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 1869 .coefficient)
      LeftAuthority1868.bound (LeftAuthority1868.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨13791⟩⟩) (rawTerms := some (Proof.Events007.exact1869RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1868.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1868.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority1868.bound []
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1868.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority1868.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound41864

namespace LeftBound41865
def owner : Owner := ⟨.program ⟨214⟩, ⟨13794⟩⟩
def transferEvent : Nat := 41865
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 41860 .summary) (.transfer 41864) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 41860 .summary)
      LeftBound41858.bound (LeftBound41858.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11312⟩⟩) (rawTerms := some (Proof.Events163.exact41860RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound41858.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 41864)
      LeftBound41864.bound (LeftBound41864.actual selector witness) := by
  exact .transfer (LeftBound41864.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound41858.bound LeftBound41864.bound
def bound : CoeffClass := .finite ⟨9984, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41858.bound, LeftBound41864.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound41858.actual selector witness) * (LeftBound41864.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound41865

namespace LeftBound41871
def owner : Owner := ⟨.program ⟨214⟩, ⟨13795⟩⟩
def transferEvent : Nat := 41871
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 41869 .coefficient) (.predecessor 1 41870 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41869 .coefficient)
      LeftAuthority1868.bound (LeftAuthority1868.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events007.exact1869RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1868.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1868.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41870 .coefficient)
      LeftBound36043.bound (LeftBound36043.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact36045RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36043.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36043.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority1868.bound LeftBound36043.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1868.bound, LeftBound36043.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority1868.actual selector witness) * (LeftBound36043.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound41871

namespace LeftBound41876
def owner : Owner := ⟨.program ⟨214⟩, ⟨7326⟩⟩
def transferEvent : Nat := 41876
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 41874 .coefficient) (.predecessor 1 41875 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41874 .coefficient)
      LeftBound35914.bound (LeftBound35914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact35915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35914.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35914.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41875 .coefficient)
      LeftBound12524.bound (LeftBound12524.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events048.exact12525RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12524.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12524.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound35914.bound LeftBound12524.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35914.bound, LeftBound12524.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound35914.actual selector witness) * (LeftBound12524.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound41876

namespace LeftBound41881
def owner : Owner := ⟨.program ⟨214⟩, ⟨13796⟩⟩
def transferEvent : Nat := 41881
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 41879 .coefficient, .predecessor 1 41880 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41879 .coefficient)
      LeftBound41876.bound (LeftBound41876.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41878RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41876.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41876.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41880 .coefficient)
      LeftBound41871.bound (LeftBound41871.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41873RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41871.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41871.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound41876.bound, LeftBound41871.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41876.bound, LeftBound41871.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound41876.actual selector witness, LeftBound41871.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41881

namespace LeftBound41885
def owner : Owner := ⟨.program ⟨214⟩, ⟨13797⟩⟩
def transferEvent : Nat := 41885
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 41883 .coefficient, .predecessor 1 41884 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41883 .coefficient)
      LeftBound41881.bound (LeftBound41881.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41882RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41881.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41881.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41884 .coefficient)
      LeftBound12516.bound (LeftBound12516.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events048.exact12517RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12516.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12516.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound41881.bound, LeftBound12516.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41881.bound, LeftBound12516.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound41881.actual selector witness, LeftBound12516.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41885

namespace LeftBound41886
def owner : Owner := ⟨.program ⟨214⟩, ⟨13797⟩⟩
def transferEvent : Nat := 41886
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨108⟩⟩]⟩ [⟨.result 12517 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12517 .coefficient)
      LeftBound12516.bound (LeftBound12516.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨108⟩⟩) (rawTerms := some (Proof.Events048.exact12517RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12516.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12516.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound12516.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12516.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound12516.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound41886

namespace LeftBound41891
def owner : Owner := ⟨.program ⟨214⟩, ⟨13798⟩⟩
def transferEvent : Nat := 41891
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 41889 .coefficient) (.predecessor 1 41890 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41889 .coefficient)
      LeftBound41885.bound (LeftBound41885.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41888RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41885.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41885.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41890 .coefficient)
      LeftBound12513.bound (LeftBound12513.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events048.exact12514RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12513.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12513.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound41885.bound LeftBound12513.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41885.bound, LeftBound12513.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound41885.actual selector witness) * (LeftBound12513.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound41891

namespace LeftBound41892
def owner : Owner := ⟨.program ⟨214⟩, ⟨13798⟩⟩
def transferEvent : Nat := 41892
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7846⟩⟩]⟩ [⟨.result 12510 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12510 .coefficient)
      LeftAuthority12509.bound (LeftAuthority12509.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7846⟩⟩) (rawTerms := some (Proof.Events048.exact12510RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12509.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12509.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority12509.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12509.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority12509.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound41892

namespace LeftBound41893
def owner : Owner := ⟨.program ⟨214⟩, ⟨13798⟩⟩
def transferEvent : Nat := 41893
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 41888 .summary) (.transfer 41892) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 41888 .summary)
      LeftBound41886.bound (LeftBound41886.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13797⟩⟩) (rawTerms := some (Proof.Events163.exact41888RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound41886.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 41892)
      LeftBound41892.bound (LeftBound41892.actual selector witness) := by
  exact .transfer (LeftBound41892.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound41886.bound LeftBound41892.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41886.bound, LeftBound41892.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound41886.actual selector witness) * (LeftBound41892.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound41893

namespace LeftBound41901
def owner : Owner := ⟨.program ⟨214⟩, ⟨13799⟩⟩
def transferEvent : Nat := 41901
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 41899 .coefficient, .predecessor 1 41900 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41899 .coefficient)
      LeftBound41891.bound (LeftBound41891.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41898RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41891.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41891.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41900 .coefficient)
      LeftBound41863.bound (LeftBound41863.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41868RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41863.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41863.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound41891.bound, LeftBound41863.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41891.bound, LeftBound41863.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound41891.actual selector witness, LeftBound41863.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41901

namespace LeftBound41903
def owner : Owner := ⟨.program ⟨214⟩, ⟨13799⟩⟩
def transferEvent : Nat := 41903
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 41898 .summary, .result 41868 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 41898 .summary)
      LeftBound41893.bound (LeftBound41893.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13798⟩⟩) (rawTerms := some (Proof.Events163.exact41898RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound41893.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 41868 .summary)
      LeftBound41865.bound (LeftBound41865.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13794⟩⟩) (rawTerms := some (Proof.Events163.exact41868RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound41865.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound41893.bound, LeftBound41865.bound]
def bound : CoeffClass := .finite ⟨95430400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41893.bound, LeftBound41865.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound41893.actual selector witness, LeftBound41865.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41903

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
