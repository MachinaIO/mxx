import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard086
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard122

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound20003
def owner : Owner := ⟨.program ⟨214⟩, ⟨15677⟩⟩
def transferEvent : Nat := 20003
def frameStart : Nat := 19930
def rule : BoundRule := .sum [.predecessor 0 20001 .coefficient, .predecessor 1 20002 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20001 .coefficient)
      LeftAuthority19999.bound (LeftAuthority19999.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events078.exact20000RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19999.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19999.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20002 .coefficient)
      LeftBound19995.bound (LeftBound19995.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events078.exact19997RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19995.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19995.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority19999.bound, LeftBound19995.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19999.bound, LeftBound19995.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority19999.actual selector witness, LeftBound19995.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20003

namespace LeftBound20007
def owner : Owner := ⟨.program ⟨214⟩, ⟨27261⟩⟩
def transferEvent : Nat := 20007
def frameStart : Nat := 19930
def rule : BoundRule := .product (.predecessor 0 20005 .coefficient) (.predecessor 1 20006 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20005 .coefficient)
      LeftBound20003.bound (LeftBound20003.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events078.exact20004RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20003.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20003.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20006 .coefficient)
      LeftAuthority19980.bound (LeftAuthority19980.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events078.exact19981RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19980.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19980.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound20003.bound LeftAuthority19980.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20003.bound, LeftAuthority19980.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound20003.actual selector witness) * (LeftAuthority19980.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound20007

namespace LeftBound20018
def owner : Owner := ⟨.program ⟨214⟩, ⟨17852⟩⟩
def transferEvent : Nat := 20018
def frameStart : Nat := 19930
def rule : BoundRule := .product (.predecessor 0 20016 .coefficient) (.predecessor 1 20017 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20016 .coefficient)
      LeftAuthority19991.bound (LeftAuthority19991.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events078.exact19992RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19991.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19991.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20017 .coefficient)
      LeftAuthority20014.bound (LeftAuthority20014.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events078.exact20015RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20014.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20014.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority19991.bound LeftAuthority20014.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19991.bound, LeftAuthority20014.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority19991.actual selector witness) * (LeftAuthority20014.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound20018

namespace LeftBound20026
def owner : Owner := ⟨.program ⟨214⟩, ⟨17853⟩⟩
def transferEvent : Nat := 20026
def frameStart : Nat := 19930
def rule : BoundRule := .sum [.predecessor 0 20024 .coefficient, .predecessor 1 20025 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20024 .coefficient)
      LeftAuthority20022.bound (LeftAuthority20022.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events078.exact20023RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20022.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20022.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20025 .coefficient)
      LeftBound20018.bound (LeftBound20018.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events078.exact20020RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20018.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20018.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority20022.bound, LeftBound20018.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority20022.bound, LeftBound20018.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority20022.actual selector witness, LeftBound20018.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20026

namespace LeftBound20030
def owner : Owner := ⟨.program ⟨214⟩, ⟨27266⟩⟩
def transferEvent : Nat := 20030
def frameStart : Nat := 19930
def rule : BoundRule := .sum [.predecessor 0 20028 .coefficient, .predecessor 1 20029 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20028 .coefficient)
      LeftBound20026.bound (LeftBound20026.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events078.exact20027RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20026.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20026.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20029 .coefficient)
      LeftBound20007.bound (LeftBound20007.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events078.exact20012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20007.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20007.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20026.bound, LeftBound20007.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20026.bound, LeftBound20007.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20026.actual selector witness, LeftBound20007.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20030

namespace LeftBound20043
def owner : Owner := ⟨.program ⟨214⟩, ⟨27263⟩⟩
def transferEvent : Nat := 20043
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 20041 .coefficient, .predecessor 1 20042 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20041 .coefficient)
      LeftBound19872.bound (LeftBound19872.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events078.exact20040RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19872.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19872.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20042 .coefficient)
      LeftBound19855.bound (LeftBound19855.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events077.exact19862RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19855.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19855.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound19872.bound, LeftBound19855.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19872.bound, LeftBound19855.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound19872.actual selector witness, LeftBound19855.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20043

namespace LeftBound20046
def owner : Owner := ⟨.program ⟨214⟩, ⟨27263⟩⟩
def transferEvent : Nat := 20046
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 20040 .summary, .result 19862 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20040 .summary)
      LeftBound19874.bound (LeftBound19874.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20915⟩⟩) (rawTerms := some (Proof.Events078.exact20040RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound19874.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 19862 .summary)
      LeftBound19857.bound (LeftBound19857.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27262⟩⟩) (rawTerms := some (Proof.Events077.exact19862RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound19857.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound19874.bound, LeftBound19857.bound]
def bound : CoeffClass := .finite ⟨1291978824159503986688, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19874.bound, LeftBound19857.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound19874.actual selector witness, LeftBound19857.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20046

namespace LeftBound20050
def owner : Owner := ⟨.program ⟨214⟩, ⟨27264⟩⟩
def transferEvent : Nat := 20050
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 20048 .coefficient) (.predecessor 1 20049 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20048 .coefficient)
      LeftBound20043.bound (LeftBound20043.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events078.exact20047RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20043.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20043.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20049 .coefficient)
      LeftBound5778.bound (LeftBound5778.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5779RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5778.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5778.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound20043.bound LeftBound5778.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20043.bound, LeftBound5778.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound20043.actual selector witness) * (LeftBound5778.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound20050

namespace LeftBound20051
def owner : Owner := ⟨.program ⟨214⟩, ⟨27264⟩⟩
def transferEvent : Nat := 20051
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩ [⟨.result 5775 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5775 .coefficient)
      LeftAuthority5774.bound (LeftAuthority5774.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6649⟩⟩) (rawTerms := some (Proof.Events022.exact5775RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5774.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5774.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5774.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5774.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5774.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound20051

namespace LeftBound20052
def owner : Owner := ⟨.program ⟨214⟩, ⟨27264⟩⟩
def transferEvent : Nat := 20052
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 20047 .summary) (.transfer 20051) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20047 .summary)
      LeftBound20046.bound (LeftBound20046.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27263⟩⟩) (rawTerms := some (Proof.Events078.exact20047RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound20046.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 20051)
      LeftBound20051.bound (LeftBound20051.actual selector witness) := by
  exact .transfer (LeftBound20051.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound20046.bound LeftBound20051.bound
def bound : CoeffClass := .finite ⟨4741582956326566183208747008, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20046.bound, LeftBound20051.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound20046.actual selector witness) * (LeftBound20051.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound20052

namespace LeftBound20067
def owner : Owner := ⟨.program ⟨214⟩, ⟨27045⟩⟩
def transferEvent : Nat := 20067
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 20065 .coefficient) (.predecessor 1 20066 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20065 .coefficient)
      LeftBound13757.bound (LeftBound13757.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events053.exact13761RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13757.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13757.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20066 .coefficient)
      LeftAuthority20063.bound (LeftAuthority20063.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events078.exact20064RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20063.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20063.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound13757.bound LeftAuthority20063.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13757.bound, LeftAuthority20063.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound13757.actual selector witness) * (LeftAuthority20063.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound20067

namespace LeftBound20068
def owner : Owner := ⟨.program ⟨214⟩, ⟨27045⟩⟩
def transferEvent : Nat := 20068
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27043⟩⟩]⟩ [⟨.result 20064 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20064 .coefficient)
      LeftAuthority20063.bound (LeftAuthority20063.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27043⟩⟩) (rawTerms := some (Proof.Events078.exact20064RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20063.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20063.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority20063.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority20063.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority20063.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound20068

namespace LeftBound20069
def owner : Owner := ⟨.program ⟨214⟩, ⟨27045⟩⟩
def transferEvent : Nat := 20069
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 13761 .summary) (.transfer 20068) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13761 .summary)
      LeftBound13760.bound (LeftBound13760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25318⟩⟩) (rawTerms := some (Proof.Events053.exact13761RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound13760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 20068)
      LeftBound20068.bound (LeftBound20068.actual selector witness) := by
  exact .transfer (LeftBound20068.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound13760.bound LeftBound20068.bound
def bound : CoeffClass := .finite ⟨1291933997458159304704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13760.bound, LeftBound20068.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound13760.actual selector witness) * (LeftBound20068.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound20069

namespace LeftBound20080
def owner : Owner := ⟨.program ⟨214⟩, ⟨20770⟩⟩
def transferEvent : Nat := 20080
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 20078 .coefficient) (.value (.predecessor 1 20079 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20078 .coefficient)
      LeftAuthority20076.bound (LeftAuthority20076.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events078.exact20077RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20076.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20076.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20079 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority20076.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority20076.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority20076.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound20080

namespace LeftBound20084
def owner : Owner := ⟨.program ⟨214⟩, ⟨20771⟩⟩
def transferEvent : Nat := 20084
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 20082 .coefficient) (.predecessor 1 20083 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20082 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20083 .coefficient)
      LeftBound20080.bound (LeftBound20080.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events078.exact20081RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20080.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20080.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound20080.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound20080.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound20080.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound20084

namespace LeftBound20085
def owner : Owner := ⟨.program ⟨214⟩, ⟨20771⟩⟩
def transferEvent : Nat := 20085
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20768⟩⟩]⟩ [⟨.result 20077 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20077 .coefficient)
      LeftAuthority20076.bound (LeftAuthority20076.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20768⟩⟩) (rawTerms := some (Proof.Events078.exact20077RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20076.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20076.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority20076.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority20076.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority20076.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound20085

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
