import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard336

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound50572
def owner : Owner := ⟨.program ⟨214⟩, ⟨18861⟩⟩
def transferEvent : Nat := 50572
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18848⟩⟩], []⟩ [⟨.result 603 .coefficient, true, some 1⟩, ⟨.result 2879 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 603 .coefficient)
      LeftAuthority602.bound (LeftAuthority602.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6490⟩⟩) (rawTerms := some (Proof.Events002.exact603RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority602.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority602.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2879 .coefficient)
      LeftAuthority2878.bound (LeftAuthority2878.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨18848⟩⟩) (rawTerms := some (Proof.Events011.exact2879RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2878.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2878.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority602.bound [LeftAuthority2878.bound]
def bound : CoeffClass := .finite ⟨228236850212900051643120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority602.bound, LeftAuthority2878.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority602.actual selector witness) * ([LeftAuthority2878.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound50572

namespace LeftBound50573
def owner : Owner := ⟨.program ⟨214⟩, ⟨18861⟩⟩
def transferEvent : Nat := 50573
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 50571, .transfer 50572]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 50571)
      LeftBound50571.bound (LeftBound50571.actual selector witness) := by
  exact .transfer (LeftBound50571.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 50572)
      LeftBound50572.bound (LeftBound50572.actual selector witness) := by
  exact .transfer (LeftBound50572.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50571.bound, LeftBound50572.bound]
def bound : CoeffClass := .finite ⟨5829664127815216198670160, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50571.bound, LeftBound50572.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50571.actual selector witness, LeftBound50572.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50573

namespace LeftBound50574
def owner : Owner := ⟨.program ⟨214⟩, ⟨18861⟩⟩
def transferEvent : Nat := 50574
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17610⟩⟩], []⟩ [⟨.result 613 .coefficient, true, some 1⟩, ⟨.result 2887 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 613 .coefficient)
      LeftAuthority612.bound (LeftAuthority612.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6494⟩⟩) (rawTerms := some (Proof.Events002.exact613RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority612.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority612.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2887 .coefficient)
      LeftAuthority2886.bound (LeftAuthority2886.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨17610⟩⟩) (rawTerms := some (Proof.Events011.exact2887RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2886.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2886.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority612.bound [LeftAuthority2886.bound]
def bound : CoeffClass := .finite ⟨227009770373045750290200, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority612.bound, LeftAuthority2886.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority612.actual selector witness) * ([LeftAuthority2886.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound50574

namespace LeftBound50575
def owner : Owner := ⟨.program ⟨214⟩, ⟨18861⟩⟩
def transferEvent : Nat := 50575
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 50573, .transfer 50574]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 50573)
      LeftBound50573.bound (LeftBound50573.actual selector witness) := by
  exact .transfer (LeftBound50573.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 50574)
      LeftBound50574.bound (LeftBound50574.actual selector witness) := by
  exact .transfer (LeftBound50574.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50573.bound, LeftBound50574.bound]
def bound : CoeffClass := .finite ⟨6056673898188261948960360, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50573.bound, LeftBound50574.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50573.actual selector witness, LeftBound50574.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50575

namespace LeftBound50576
def owner : Owner := ⟨.program ⟨214⟩, ⟨18861⟩⟩
def transferEvent : Nat := 50576
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17666⟩⟩], []⟩ [⟨.result 623 .coefficient, true, some 1⟩, ⟨.result 2895 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 623 .coefficient)
      LeftAuthority622.bound (LeftAuthority622.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6502⟩⟩) (rawTerms := some (Proof.Events002.exact623RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority622.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority622.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2895 .coefficient)
      LeftAuthority2894.bound (LeftAuthority2894.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨17666⟩⟩) (rawTerms := some (Proof.Events011.exact2895RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2894.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2894.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority622.bound [LeftAuthority2894.bound]
def bound : CoeffClass := .finite ⟨226487908831958288795280, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority622.bound, LeftAuthority2894.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority622.actual selector witness) * ([LeftAuthority2894.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound50576

namespace LeftBound50577
def owner : Owner := ⟨.program ⟨214⟩, ⟨18861⟩⟩
def transferEvent : Nat := 50577
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 50575, .transfer 50576]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 50575)
      LeftBound50575.bound (LeftBound50575.actual selector witness) := by
  exact .transfer (LeftBound50575.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 50576)
      LeftBound50576.bound (LeftBound50576.actual selector witness) := by
  exact .transfer (LeftBound50576.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50575.bound, LeftBound50576.bound]
def bound : CoeffClass := .finite ⟨6283161807020220237755640, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50575.bound, LeftBound50576.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50575.actual selector witness, LeftBound50576.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50577

namespace LeftBound50578
def owner : Owner := ⟨.program ⟨214⟩, ⟨18861⟩⟩
def transferEvent : Nat := 50578
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18042⟩⟩], []⟩ [⟨.result 633 .coefficient, true, some 1⟩, ⟨.result 2903 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 633 .coefficient)
      LeftAuthority632.bound (LeftAuthority632.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6383⟩⟩) (rawTerms := some (Proof.Events002.exact633RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority632.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority632.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2903 .coefficient)
      LeftAuthority2902.bound (LeftAuthority2902.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨18042⟩⟩) (rawTerms := some (Proof.Events011.exact2903RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2902.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2902.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority632.bound [LeftAuthority2902.bound]
def bound : CoeffClass := .finite ⟨224377773035387248837560, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority632.bound, LeftAuthority2902.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority632.actual selector witness) * ([LeftAuthority2902.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound50578

namespace LeftBound50579
def owner : Owner := ⟨.program ⟨214⟩, ⟨18861⟩⟩
def transferEvent : Nat := 50579
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 50577, .transfer 50578]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 50577)
      LeftBound50577.bound (LeftBound50577.actual selector witness) := by
  exact .transfer (LeftBound50577.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 50578)
      LeftBound50578.bound (LeftBound50578.actual selector witness) := by
  exact .transfer (LeftBound50578.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50577.bound, LeftBound50578.bound]
def bound : CoeffClass := .finite ⟨6507539580055607486593200, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50577.bound, LeftBound50578.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50577.actual selector witness, LeftBound50578.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50579

namespace LeftBound50580
def owner : Owner := ⟨.program ⟨214⟩, ⟨18861⟩⟩
def transferEvent : Nat := 50580
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17169⟩⟩], []⟩ [⟨.result 643 .coefficient, true, some 1⟩, ⟨.result 2911 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 643 .coefficient)
      LeftAuthority642.bound (LeftAuthority642.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6387⟩⟩) (rawTerms := some (Proof.Events002.exact643RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority642.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority642.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2911 .coefficient)
      LeftAuthority2910.bound (LeftAuthority2910.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨17169⟩⟩) (rawTerms := some (Proof.Events011.exact2911RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2910.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2910.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority642.bound [LeftAuthority2910.bound]
def bound : CoeffClass := .finite ⟨222230617312560576599880, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority642.bound, LeftAuthority2910.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority642.actual selector witness) * ([LeftAuthority2910.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound50580

namespace LeftBound50581
def owner : Owner := ⟨.program ⟨214⟩, ⟨18861⟩⟩
def transferEvent : Nat := 50581
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 50579, .transfer 50580]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 50579)
      LeftBound50579.bound (LeftBound50579.actual selector witness) := by
  exact .transfer (LeftBound50579.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 50580)
      LeftBound50580.bound (LeftBound50580.actual selector witness) := by
  exact .transfer (LeftBound50580.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50579.bound, LeftBound50580.bound]
def bound : CoeffClass := .finite ⟨6729770197368168063193080, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50579.bound, LeftBound50580.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50579.actual selector witness, LeftBound50580.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50581

namespace LeftBound50582
def owner : Owner := ⟨.program ⟨214⟩, ⟨18861⟩⟩
def transferEvent : Nat := 50582
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17225⟩⟩], []⟩ [⟨.result 653 .coefficient, true, some 1⟩, ⟨.result 2919 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 653 .coefficient)
      LeftAuthority652.bound (LeftAuthority652.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6391⟩⟩) (rawTerms := some (Proof.Events002.exact653RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority652.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority652.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2919 .coefficient)
      LeftAuthority2918.bound (LeftAuthority2918.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨17225⟩⟩) (rawTerms := some (Proof.Events011.exact2919RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2918.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2918.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority652.bound [LeftAuthority2918.bound]
def bound : CoeffClass := .finite ⟨220778129617707239497920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority652.bound, LeftAuthority2918.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority652.actual selector witness) * ([LeftAuthority2918.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound50582

namespace LeftBound50583
def owner : Owner := ⟨.program ⟨214⟩, ⟨18861⟩⟩
def transferEvent : Nat := 50583
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 50581, .transfer 50582]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 50581)
      LeftBound50581.bound (LeftBound50581.actual selector witness) := by
  exact .transfer (LeftBound50581.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 50582)
      LeftBound50582.bound (LeftBound50582.actual selector witness) := by
  exact .transfer (LeftBound50582.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50581.bound, LeftBound50582.bound]
def bound : CoeffClass := .finite ⟨6950548326985875302691000, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50581.bound, LeftBound50582.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50581.actual selector witness, LeftBound50582.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50583

namespace LeftBound50584
def owner : Owner := ⟨.program ⟨214⟩, ⟨18861⟩⟩
def transferEvent : Nat := 50584
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17442⟩⟩], []⟩ [⟨.result 663 .coefficient, true, some 1⟩, ⟨.result 2927 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 663 .coefficient)
      LeftAuthority662.bound (LeftAuthority662.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6398⟩⟩) (rawTerms := some (Proof.Events002.exact663RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority662.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority662.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2927 .coefficient)
      LeftAuthority2926.bound (LeftAuthority2926.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨17442⟩⟩) (rawTerms := some (Proof.Events011.exact2927RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2926.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2926.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority662.bound [LeftAuthority2926.bound]
def bound : CoeffClass := .finite ⟨216532396355828254122960, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority662.bound, LeftAuthority2926.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority662.actual selector witness) * ([LeftAuthority2926.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound50584

namespace LeftBound50585
def owner : Owner := ⟨.program ⟨214⟩, ⟨18861⟩⟩
def transferEvent : Nat := 50585
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 50583, .transfer 50584]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 50583)
      LeftBound50583.bound (LeftBound50583.actual selector witness) := by
  exact .transfer (LeftBound50583.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 50584)
      LeftBound50584.bound (LeftBound50584.actual selector witness) := by
  exact .transfer (LeftBound50584.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50583.bound, LeftBound50584.bound]
def bound : CoeffClass := .finite ⟨7167080723341703556813960, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50583.bound, LeftBound50584.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50583.actual selector witness, LeftBound50584.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50585

namespace LeftBound50586
def owner : Owner := ⟨.program ⟨214⟩, ⟨18861⟩⟩
def transferEvent : Nat := 50586
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17822⟩⟩], []⟩ [⟨.result 673 .coefficient, true, some 1⟩, ⟨.result 2935 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 673 .coefficient)
      LeftAuthority672.bound (LeftAuthority672.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6407⟩⟩) (rawTerms := some (Proof.Events002.exact673RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority672.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority672.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2935 .coefficient)
      LeftAuthority2934.bound (LeftAuthority2934.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨17822⟩⟩) (rawTerms := some (Proof.Events011.exact2935RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2934.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2934.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority672.bound [LeftAuthority2934.bound]
def bound : CoeffClass := .finite ⟨213251602471649038151400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority672.bound, LeftAuthority2934.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority672.actual selector witness) * ([LeftAuthority2934.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound50586

namespace LeftBound50587
def owner : Owner := ⟨.program ⟨214⟩, ⟨18861⟩⟩
def transferEvent : Nat := 50587
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 50585, .transfer 50586]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 50585)
      LeftBound50585.bound (LeftBound50585.actual selector witness) := by
  exact .transfer (LeftBound50585.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 50586)
      LeftBound50586.bound (LeftBound50586.actual selector witness) := by
  exact .transfer (LeftBound50586.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50585.bound, LeftBound50586.bound]
def bound : CoeffClass := .finite ⟨7380332325813352594965360, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50585.bound, LeftBound50586.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50585.actual selector witness, LeftBound50586.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50587

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
