import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard032
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard033
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard234
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard237

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound36541
def owner : Owner := ⟨.program ⟨214⟩, ⟨13173⟩⟩
def transferEvent : Nat := 36541
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 36539 .coefficient) (.predecessor 1 36540 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36539 .coefficient)
      LeftAuthority1612.bound (LeftAuthority1612.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events006.exact1613RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1612.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1612.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36540 .coefficient)
      LeftBound36043.bound (LeftBound36043.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact36045RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36043.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36043.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority1612.bound LeftBound36043.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1612.bound, LeftBound36043.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority1612.actual selector witness) * (LeftBound36043.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound36541

namespace LeftBound36546
def owner : Owner := ⟨.program ⟨214⟩, ⟨7321⟩⟩
def transferEvent : Nat := 36546
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 36544 .coefficient) (.predecessor 1 36545 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36544 .coefficient)
      LeftBound35914.bound (LeftBound35914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact35915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35914.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35914.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36545 .coefficient)
      LeftBound6972.bound (LeftBound6972.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact6973RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6972.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6972.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound35914.bound LeftBound6972.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35914.bound, LeftBound6972.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound35914.actual selector witness) * (LeftBound6972.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36546

namespace LeftBound36551
def owner : Owner := ⟨.program ⟨214⟩, ⟨13174⟩⟩
def transferEvent : Nat := 36551
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 36549 .coefficient, .predecessor 1 36550 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36549 .coefficient)
      LeftBound36546.bound (LeftBound36546.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events142.exact36548RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36546.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36546.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36550 .coefficient)
      LeftBound36541.bound (LeftBound36541.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events142.exact36543RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36541.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36541.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound36546.bound, LeftBound36541.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36546.bound, LeftBound36541.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound36546.actual selector witness, LeftBound36541.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound36551

namespace LeftBound36555
def owner : Owner := ⟨.program ⟨214⟩, ⟨13175⟩⟩
def transferEvent : Nat := 36555
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 36553 .coefficient, .predecessor 1 36554 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36553 .coefficient)
      LeftBound36551.bound (LeftBound36551.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events142.exact36552RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36551.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36551.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36554 .coefficient)
      LeftBound6964.bound (LeftBound6964.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact6965RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6964.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6964.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound36551.bound, LeftBound6964.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36551.bound, LeftBound6964.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound36551.actual selector witness, LeftBound6964.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound36555

namespace LeftBound36556
def owner : Owner := ⟨.program ⟨214⟩, ⟨13175⟩⟩
def transferEvent : Nat := 36556
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨103⟩⟩]⟩ [⟨.result 6965 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6965 .coefficient)
      LeftBound6964.bound (LeftBound6964.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨103⟩⟩) (rawTerms := some (Proof.Events027.exact6965RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6964.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6964.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound6964.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6964.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound6964.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound36556

namespace LeftBound36561
def owner : Owner := ⟨.program ⟨214⟩, ⟨13176⟩⟩
def transferEvent : Nat := 36561
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 36559 .coefficient) (.predecessor 1 36560 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36559 .coefficient)
      LeftBound36555.bound (LeftBound36555.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events142.exact36558RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36555.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36555.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36560 .coefficient)
      LeftAuthority1615.bound (LeftAuthority1615.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events006.exact1616RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1615.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1615.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound36555.bound LeftAuthority1615.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36555.bound, LeftAuthority1615.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound36555.actual selector witness) * (LeftAuthority1615.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36561

namespace LeftBound36562
def owner : Owner := ⟨.program ⟨214⟩, ⟨13176⟩⟩
def transferEvent : Nat := 36562
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨10250⟩⟩], []⟩ [⟨.result 1616 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 1616 .coefficient)
      LeftAuthority1615.bound (LeftAuthority1615.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨10250⟩⟩) (rawTerms := some (Proof.Events006.exact1616RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1615.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1615.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority1615.bound []
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1615.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority1615.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound36562

namespace LeftBound36563
def owner : Owner := ⟨.program ⟨214⟩, ⟨13176⟩⟩
def transferEvent : Nat := 36563
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36558 .summary) (.transfer 36562) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36558 .summary)
      LeftBound36556.bound (LeftBound36556.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13175⟩⟩) (rawTerms := some (Proof.Events142.exact36558RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36556.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 36562)
      LeftBound36562.bound (LeftBound36562.actual selector witness) := by
  exact .transfer (LeftBound36562.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound36556.bound LeftBound36562.bound
def bound : CoeffClass := .finite ⟨48256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36556.bound, LeftBound36562.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound36556.actual selector witness) * (LeftBound36562.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36563

namespace LeftBound36569
def owner : Owner := ⟨.program ⟨214⟩, ⟨10251⟩⟩
def transferEvent : Nat := 36569
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 36567 .coefficient) (.predecessor 1 36568 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36567 .coefficient)
      LeftAuthority1615.bound (LeftAuthority1615.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events006.exact1616RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1615.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1615.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36568 .coefficient)
      LeftBound36043.bound (LeftBound36043.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact36045RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36043.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36043.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority1615.bound LeftBound36043.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1615.bound, LeftBound36043.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority1615.actual selector witness) * (LeftBound36043.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound36569

namespace LeftBound36574
def owner : Owner := ⟨.program ⟨214⟩, ⟨7301⟩⟩
def transferEvent : Nat := 36574
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 36572 .coefficient) (.predecessor 1 36573 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36572 .coefficient)
      LeftBound35914.bound (LeftBound35914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact35915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35914.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35914.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36573 .coefficient)
      LeftBound7013.bound (LeftBound7013.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact7014RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7013.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7013.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound35914.bound LeftBound7013.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35914.bound, LeftBound7013.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound35914.actual selector witness) * (LeftBound7013.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36574

namespace LeftBound36579
def owner : Owner := ⟨.program ⟨214⟩, ⟨10252⟩⟩
def transferEvent : Nat := 36579
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 36577 .coefficient, .predecessor 1 36578 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36577 .coefficient)
      LeftBound36574.bound (LeftBound36574.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events142.exact36576RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36574.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36574.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36578 .coefficient)
      LeftBound36569.bound (LeftBound36569.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events142.exact36571RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36569.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36569.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound36574.bound, LeftBound36569.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36574.bound, LeftBound36569.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound36574.actual selector witness, LeftBound36569.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound36579

namespace LeftBound36583
def owner : Owner := ⟨.program ⟨214⟩, ⟨10253⟩⟩
def transferEvent : Nat := 36583
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 36581 .coefficient, .predecessor 1 36582 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36581 .coefficient)
      LeftBound36579.bound (LeftBound36579.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events142.exact36580RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36579.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36579.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36582 .coefficient)
      LeftBound7005.bound (LeftBound7005.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact7006RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7005.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7005.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound36579.bound, LeftBound7005.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36579.bound, LeftBound7005.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound36579.actual selector witness, LeftBound7005.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound36583

namespace LeftBound36584
def owner : Owner := ⟨.program ⟨214⟩, ⟨10253⟩⟩
def transferEvent : Nat := 36584
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨83⟩⟩]⟩ [⟨.result 7006 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7006 .coefficient)
      LeftBound7005.bound (LeftBound7005.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨83⟩⟩) (rawTerms := some (Proof.Events027.exact7006RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7005.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7005.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound7005.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7005.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound7005.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound36584

namespace LeftBound36589
def owner : Owner := ⟨.program ⟨214⟩, ⟨10254⟩⟩
def transferEvent : Nat := 36589
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 36587 .coefficient) (.predecessor 1 36588 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36587 .coefficient)
      LeftBound36583.bound (LeftBound36583.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events142.exact36586RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36583.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36583.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36588 .coefficient)
      LeftBound7002.bound (LeftBound7002.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact7003RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7002.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7002.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound36583.bound LeftBound7002.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36583.bound, LeftBound7002.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound36583.actual selector witness) * (LeftBound7002.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36589

namespace LeftBound36590
def owner : Owner := ⟨.program ⟨214⟩, ⟨10254⟩⟩
def transferEvent : Nat := 36590
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩ [⟨.result 6999 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6999 .coefficient)
      LeftAuthority6998.bound (LeftAuthority6998.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7879⟩⟩) (rawTerms := some (Proof.Events027.exact6999RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6998.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6998.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority6998.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6998.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority6998.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound36590

namespace LeftBound36591
def owner : Owner := ⟨.program ⟨214⟩, ⟨10254⟩⟩
def transferEvent : Nat := 36591
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36586 .summary) (.transfer 36590) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36586 .summary)
      LeftBound36584.bound (LeftBound36584.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10253⟩⟩) (rawTerms := some (Proof.Events142.exact36586RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36584.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 36590)
      LeftBound36590.bound (LeftBound36590.actual selector witness) := by
  exact .transfer (LeftBound36590.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound36584.bound LeftBound36590.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36584.bound, LeftBound36590.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound36584.actual selector witness) * (LeftBound36590.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36591

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
