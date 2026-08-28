import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard044
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard045
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard538
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard541
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard555

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound81814
def owner : Owner := ⟨.program ⟨214⟩, ⟨29390⟩⟩
def transferEvent : Nat := 81814
def frameStart : Nat := 81714
def rule : BoundRule := .sum [.predecessor 0 81812 .coefficient, .predecessor 1 81813 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81812 .coefficient)
      LeftBound81810.bound (LeftBound81810.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events319.exact81811RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81810.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81810.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81813 .coefficient)
      LeftBound81791.bound (LeftBound81791.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events319.exact81796RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81791.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81791.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound81810.bound, LeftBound81791.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81810.bound, LeftBound81791.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound81810.actual selector witness, LeftBound81791.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound81814

namespace LeftBound81827
def owner : Owner := ⟨.program ⟨214⟩, ⟨29388⟩⟩
def transferEvent : Nat := 81827
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 81825 .coefficient, .predecessor 1 81826 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81825 .coefficient)
      LeftBound81656.bound (LeftBound81656.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events319.exact81824RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81656.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81656.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81826 .coefficient)
      LeftBound81639.bound (LeftBound81639.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events318.exact81646RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81639.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81639.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound81656.bound, LeftBound81639.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81656.bound, LeftBound81639.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound81656.actual selector witness, LeftBound81639.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound81827

namespace LeftBound81830
def owner : Owner := ⟨.program ⟨214⟩, ⟨29388⟩⟩
def transferEvent : Nat := 81830
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 81824 .summary, .result 81646 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 81824 .summary)
      LeftBound81658.bound (LeftBound81658.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22411⟩⟩) (rawTerms := some (Proof.Events319.exact81824RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound81658.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 81646 .summary)
      LeftBound81641.bound (LeftBound81641.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29387⟩⟩) (rawTerms := some (Proof.Events318.exact81646RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound81641.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound81658.bound, LeftBound81641.bound]
def bound : CoeffClass := .finite ⟨1292382248169874534400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81658.bound, LeftBound81641.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound81658.actual selector witness, LeftBound81641.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound81830

namespace LeftBound81854
def owner : Owner := ⟨.program ⟨214⟩, ⟨12569⟩⟩
def transferEvent : Nat := 81854
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 81852 .coefficient) (.predecessor 1 81853 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81852 .coefficient)
      LeftAuthority3919.bound (LeftAuthority3919.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events015.exact3920RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3919.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3919.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81853 .coefficient)
      LeftBound79918.bound (LeftBound79918.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79920RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79918.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79918.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority3919.bound LeftBound79918.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3919.bound, LeftBound79918.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority3919.actual selector witness) * (LeftBound79918.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound81854

namespace LeftBound81859
def owner : Owner := ⟨.program ⟨214⟩, ⟨7242⟩⟩
def transferEvent : Nat := 81859
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 81857 .coefficient) (.predecessor 1 81858 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81857 .coefficient)
      LeftBound79789.bound (LeftBound79789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events311.exact79790RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79789.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79789.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81858 .coefficient)
      LeftBound8475.bound (LeftBound8475.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events033.exact8476RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8475.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8475.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound79789.bound LeftBound8475.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79789.bound, LeftBound8475.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound79789.actual selector witness) * (LeftBound8475.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound81859

namespace LeftBound81864
def owner : Owner := ⟨.program ⟨214⟩, ⟨12570⟩⟩
def transferEvent : Nat := 81864
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 81862 .coefficient, .predecessor 1 81863 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81862 .coefficient)
      LeftBound81859.bound (LeftBound81859.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events319.exact81861RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81859.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81859.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81863 .coefficient)
      LeftBound81854.bound (LeftBound81854.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events319.exact81856RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81854.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81854.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound81859.bound, LeftBound81854.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81859.bound, LeftBound81854.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound81859.actual selector witness, LeftBound81854.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound81864

namespace LeftBound81868
def owner : Owner := ⟨.program ⟨214⟩, ⟨12571⟩⟩
def transferEvent : Nat := 81868
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 81866 .coefficient, .predecessor 1 81867 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81866 .coefficient)
      LeftBound81864.bound (LeftBound81864.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events319.exact81865RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81864.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81864.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81867 .coefficient)
      LeftBound8467.bound (LeftBound8467.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events033.exact8468RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8467.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8467.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound81864.bound, LeftBound8467.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81864.bound, LeftBound8467.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound81864.actual selector witness, LeftBound8467.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound81868

namespace LeftBound81869
def owner : Owner := ⟨.program ⟨214⟩, ⟨12571⟩⟩
def transferEvent : Nat := 81869
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨100⟩⟩]⟩ [⟨.result 8468 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 8468 .coefficient)
      LeftBound8467.bound (LeftBound8467.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨100⟩⟩) (rawTerms := some (Proof.Events033.exact8468RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8467.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8467.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound8467.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8467.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound8467.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound81869

namespace LeftBound81874
def owner : Owner := ⟨.program ⟨214⟩, ⟨12572⟩⟩
def transferEvent : Nat := 81874
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 81872 .coefficient) (.predecessor 1 81873 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81872 .coefficient)
      LeftBound81868.bound (LeftBound81868.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events319.exact81871RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81868.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81868.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81873 .coefficient)
      LeftAuthority3922.bound (LeftAuthority3922.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events015.exact3923RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3922.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3922.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound81868.bound LeftAuthority3922.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81868.bound, LeftAuthority3922.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound81868.actual selector witness) * (LeftAuthority3922.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound81874

namespace LeftBound81875
def owner : Owner := ⟨.program ⟨214⟩, ⟨12572⟩⟩
def transferEvent : Nat := 81875
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨9925⟩⟩], []⟩ [⟨.result 3923 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 3923 .coefficient)
      LeftAuthority3922.bound (LeftAuthority3922.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨9925⟩⟩) (rawTerms := some (Proof.Events015.exact3923RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3922.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3922.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority3922.bound []
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3922.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority3922.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound81875

namespace LeftBound81876
def owner : Owner := ⟨.program ⟨214⟩, ⟨12572⟩⟩
def transferEvent : Nat := 81876
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 81871 .summary) (.transfer 81875) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 81871 .summary)
      LeftBound81869.bound (LeftBound81869.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12571⟩⟩) (rawTerms := some (Proof.Events319.exact81871RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound81869.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 81875)
      LeftBound81875.bound (LeftBound81875.actual selector witness) := by
  exact .transfer (LeftBound81875.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound81869.bound LeftBound81875.bound
def bound : CoeffClass := .finite ⟨34944, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81869.bound, LeftBound81875.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound81869.actual selector witness) * (LeftBound81875.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound81876

namespace LeftBound81882
def owner : Owner := ⟨.program ⟨214⟩, ⟨9926⟩⟩
def transferEvent : Nat := 81882
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 81880 .coefficient) (.predecessor 1 81881 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81880 .coefficient)
      LeftAuthority3922.bound (LeftAuthority3922.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events015.exact3923RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3922.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3922.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81881 .coefficient)
      LeftBound79918.bound (LeftBound79918.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79920RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79918.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79918.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority3922.bound LeftBound79918.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3922.bound, LeftBound79918.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority3922.actual selector witness) * (LeftBound79918.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound81882

namespace LeftBound81887
def owner : Owner := ⟨.program ⟨214⟩, ⟨7222⟩⟩
def transferEvent : Nat := 81887
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 81885 .coefficient) (.predecessor 1 81886 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81885 .coefficient)
      LeftBound79789.bound (LeftBound79789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events311.exact79790RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79789.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79789.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81886 .coefficient)
      LeftBound8516.bound (LeftBound8516.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events033.exact8517RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8516.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8516.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound79789.bound LeftBound8516.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79789.bound, LeftBound8516.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound79789.actual selector witness) * (LeftBound8516.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound81887

namespace LeftBound81892
def owner : Owner := ⟨.program ⟨214⟩, ⟨9927⟩⟩
def transferEvent : Nat := 81892
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 81890 .coefficient, .predecessor 1 81891 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81890 .coefficient)
      LeftBound81887.bound (LeftBound81887.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events319.exact81889RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81887.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81887.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81891 .coefficient)
      LeftBound81882.bound (LeftBound81882.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events319.exact81884RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81882.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81882.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound81887.bound, LeftBound81882.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81887.bound, LeftBound81882.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound81887.actual selector witness, LeftBound81882.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound81892

namespace LeftBound81896
def owner : Owner := ⟨.program ⟨214⟩, ⟨9928⟩⟩
def transferEvent : Nat := 81896
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 81894 .coefficient, .predecessor 1 81895 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81894 .coefficient)
      LeftBound81892.bound (LeftBound81892.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events319.exact81893RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81892.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81892.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81895 .coefficient)
      LeftBound8508.bound (LeftBound8508.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events033.exact8509RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8508.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8508.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound81892.bound, LeftBound8508.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81892.bound, LeftBound8508.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound81892.actual selector witness, LeftBound8508.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound81896

namespace LeftBound81897
def owner : Owner := ⟨.program ⟨214⟩, ⟨9928⟩⟩
def transferEvent : Nat := 81897
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨80⟩⟩]⟩ [⟨.result 8509 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 8509 .coefficient)
      LeftBound8508.bound (LeftBound8508.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨80⟩⟩) (rawTerms := some (Proof.Events033.exact8509RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8508.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8508.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound8508.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8508.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound8508.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound81897

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
