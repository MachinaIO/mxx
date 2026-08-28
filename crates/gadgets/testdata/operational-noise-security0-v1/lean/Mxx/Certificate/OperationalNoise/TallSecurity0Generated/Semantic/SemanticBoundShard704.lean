import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard702
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard703

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound101973
def owner : Owner := ⟨.program ⟨214⟩, ⟨14784⟩⟩
def transferEvent : Nat := 101973
def frameStart : Nat := 101883
def rule : BoundRule := .product (.predecessor 0 101971 .coefficient) (.predecessor 1 101972 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101971 .coefficient)
      LeftAuthority101926.bound (LeftAuthority101926.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events398.exact101927RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101926.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101926.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101972 .coefficient)
      LeftAuthority101969.bound (LeftAuthority101969.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events398.exact101970RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101969.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101969.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority101926.bound LeftAuthority101969.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority101926.bound, LeftAuthority101969.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority101926.actual selector witness) * (LeftAuthority101969.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound101973

namespace LeftBound101981
def owner : Owner := ⟨.program ⟨214⟩, ⟨14785⟩⟩
def transferEvent : Nat := 101981
def frameStart : Nat := 101883
def rule : BoundRule := .sum [.predecessor 0 101979 .coefficient, .predecessor 1 101980 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101979 .coefficient)
      LeftAuthority101977.bound (LeftAuthority101977.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events398.exact101978RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101977.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101977.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101980 .coefficient)
      LeftBound101973.bound (LeftBound101973.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events398.exact101975RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101973.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101973.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority101977.bound, LeftBound101973.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority101977.bound, LeftBound101973.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority101977.actual selector witness, LeftBound101973.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound101981

namespace LeftBound101985
def owner : Owner := ⟨.program ⟨214⟩, ⟨24902⟩⟩
def transferEvent : Nat := 101985
def frameStart : Nat := 101883
def rule : BoundRule := .sum [.predecessor 0 101983 .coefficient, .predecessor 1 101984 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101983 .coefficient)
      LeftBound101981.bound (LeftBound101981.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events398.exact101982RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101981.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101981.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101984 .coefficient)
      LeftBound101962.bound (LeftBound101962.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events398.exact101967RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101962.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101962.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound101981.bound, LeftBound101962.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101981.bound, LeftBound101962.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound101981.actual selector witness, LeftBound101962.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound101985

namespace LeftBound101998
def owner : Owner := ⟨.program ⟨214⟩, ⟨24900⟩⟩
def transferEvent : Nat := 101998
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 101996 .coefficient, .predecessor 1 101997 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101996 .coefficient)
      LeftBound101843.bound (LeftBound101843.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events398.exact101995RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101843.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101843.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101997 .coefficient)
      LeftBound101826.bound (LeftBound101826.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events397.exact101833RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101826.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101826.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound101843.bound, LeftBound101826.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101843.bound, LeftBound101826.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound101843.actual selector witness, LeftBound101826.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound101998

namespace LeftBound102001
def owner : Owner := ⟨.program ⟨214⟩, ⟨24900⟩⟩
def transferEvent : Nat := 102001
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 101995 .summary, .result 101833 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 101995 .summary)
      LeftBound101845.bound (LeftBound101845.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19016⟩⟩) (rawTerms := some (Proof.Events398.exact101995RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound101845.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 101833 .summary)
      LeftBound101828.bound (LeftBound101828.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨24899⟩⟩) (rawTerms := some (Proof.Events397.exact101833RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound101828.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound101845.bound, LeftBound101828.bound]
def bound : CoeffClass := .finite ⟨352011863863296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101845.bound, LeftBound101828.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound101845.actual selector witness, LeftBound101828.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102001

namespace LeftBound102005
def owner : Owner := ⟨.program ⟨214⟩, ⟨26328⟩⟩
def transferEvent : Nat := 102005
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 102003 .coefficient) (.predecessor 1 102004 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 102003 .coefficient)
      LeftBound101998.bound (LeftBound101998.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events398.exact102002RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101998.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101998.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 102004 .coefficient)
      LeftAuthority101748.bound (LeftAuthority101748.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events397.exact101749RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101748.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101748.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound101998.bound LeftAuthority101748.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101998.bound, LeftAuthority101748.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound101998.actual selector witness) * (LeftAuthority101748.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound102005

namespace LeftBound102006
def owner : Owner := ⟨.program ⟨214⟩, ⟨26328⟩⟩
def transferEvent : Nat := 102006
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26326⟩⟩]⟩ [⟨.result 101749 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 101749 .coefficient)
      LeftAuthority101748.bound (LeftAuthority101748.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26326⟩⟩) (rawTerms := some (Proof.Events397.exact101749RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101748.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101748.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority101748.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority101748.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority101748.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound102006

namespace LeftBound102007
def owner : Owner := ⟨.program ⟨214⟩, ⟨26328⟩⟩
def transferEvent : Nat := 102007
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 102002 .summary) (.transfer 102006) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 102002 .summary)
      LeftBound102001.bound (LeftBound102001.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨24900⟩⟩) (rawTerms := some (Proof.Events398.exact102002RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound102001.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 102006)
      LeftBound102006.bound (LeftBound102006.actual selector witness) := by
  exact .transfer (LeftBound102006.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound102001.bound LeftBound102006.bound
def bound : CoeffClass := .finite ⟨1291889172568118132736, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102001.bound, LeftBound102006.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound102001.actual selector witness) * (LeftBound102006.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound102007

namespace LeftBound102018
def owner : Owner := ⟨.program ⟨214⟩, ⟨20383⟩⟩
def transferEvent : Nat := 102018
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 102016 .coefficient) (.value (.predecessor 1 102017 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 102016 .coefficient)
      LeftAuthority102014.bound (LeftAuthority102014.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events398.exact102015RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority102014.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority102014.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 102017 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority102014.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority102014.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority102014.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound102018

namespace LeftBound102022
def owner : Owner := ⟨.program ⟨214⟩, ⟨20384⟩⟩
def transferEvent : Nat := 102022
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 102020 .coefficient) (.predecessor 1 102021 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 102020 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 102021 .coefficient)
      LeftBound102018.bound (LeftBound102018.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events398.exact102019RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102018.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102018.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound102018.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound102018.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound102018.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound102022

namespace LeftBound102023
def owner : Owner := ⟨.program ⟨214⟩, ⟨20384⟩⟩
def transferEvent : Nat := 102023
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20381⟩⟩]⟩ [⟨.result 102015 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 102015 .coefficient)
      LeftAuthority102014.bound (LeftAuthority102014.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20381⟩⟩) (rawTerms := some (Proof.Events398.exact102015RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority102014.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority102014.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority102014.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority102014.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority102014.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound102023

namespace LeftBound102024
def owner : Owner := ⟨.program ⟨214⟩, ⟨20384⟩⟩
def transferEvent : Nat := 102024
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 102023) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 102023)
      LeftBound102023.bound (LeftBound102023.actual selector witness) := by
  exact .transfer (LeftBound102023.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound102023.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound102023.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound102023.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound102024

namespace LeftBound102095
def owner : Owner := ⟨.program ⟨214⟩, ⟨14783⟩⟩
def transferEvent : Nat := 102095
def frameStart : Nat := 102068
def rule : BoundRule := .identity (.predecessor 0 102094 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 102094 .coefficient)
      LeftAuthority102092.bound (LeftAuthority102092.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events398.exact102093RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority102092.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority102092.derived selector witness)

def rawBound : CoeffClass := LeftAuthority102092.bound
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority102092.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority102092.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound102095

namespace LeftBound102112
def owner : Owner := ⟨.program ⟨214⟩, ⟨14824⟩⟩
def transferEvent : Nat := 102112
def frameStart : Nat := 102068
def rule : BoundRule := .sum [.predecessor 0 102110 .coefficient, .predecessor 1 102111 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 102110 .coefficient)
      LeftBound102095.bound (LeftBound102095.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound102095.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 102111 .coefficient)
      LeftAuthority102108.bound (LeftAuthority102108.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority102108.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102095.bound, LeftAuthority102108.bound]
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102095.bound, LeftAuthority102108.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound102095.actual selector witness, LeftAuthority102108.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102112

namespace LeftBound102115
def owner : Owner := ⟨.program ⟨214⟩, ⟨14825⟩⟩
def transferEvent : Nat := 102115
def frameStart : Nat := 102068
def rule : BoundRule := .identity (.predecessor 0 102114 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 102114 .coefficient)
      LeftBound102112.bound (LeftBound102112.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound102112.derived selector witness)

def rawBound : CoeffClass := LeftBound102112.bound
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102112.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound102112.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound102115

namespace LeftBound102121
def owner : Owner := ⟨.program ⟨214⟩, ⟨14826⟩⟩
def transferEvent : Nat := 102121
def frameStart : Nat := 102068
def rule : BoundRule := .product (.predecessor 0 102119 .coefficient) (.predecessor 1 102120 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 102119 .coefficient)
      LeftAuthority102117.bound (LeftAuthority102117.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events398.exact102118RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority102117.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority102117.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 102120 .coefficient)
      LeftBound102115.bound (LeftBound102115.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events398.exact102116RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102115.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102115.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority102117.bound LeftBound102115.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority102117.bound, LeftBound102115.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority102117.actual selector witness) * (LeftBound102115.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound102121

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
