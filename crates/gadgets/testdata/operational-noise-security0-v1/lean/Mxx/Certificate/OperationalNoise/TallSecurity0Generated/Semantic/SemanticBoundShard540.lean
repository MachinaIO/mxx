import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard539

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound79824
def owner : Owner := ⟨.program ⟨214⟩, ⟨18846⟩⟩
def transferEvent : Nat := 79824
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17606⟩⟩], []⟩ [⟨.result 613 .coefficient, true, some 1⟩, ⟨.result 4377 .coefficient, true, some 1⟩]
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
    BoundInputAt history owner (.result 4377 .coefficient)
      LeftAuthority4376.bound (LeftAuthority4376.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨17606⟩⟩) (rawTerms := some (Proof.Events017.exact4377RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4376.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4376.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority612.bound [LeftAuthority4376.bound]
def bound : CoeffClass := .finite ⟨227009770373045750290200, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority612.bound, LeftAuthority4376.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority612.actual selector witness) * ([LeftAuthority4376.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound79824

namespace LeftBound79825
def owner : Owner := ⟨.program ⟨214⟩, ⟨18846⟩⟩
def transferEvent : Nat := 79825
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 79823, .transfer 79824]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79823)
      LeftBound79823.bound (LeftBound79823.actual selector witness) := by
  exact .transfer (LeftBound79823.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79824)
      LeftBound79824.bound (LeftBound79824.actual selector witness) := by
  exact .transfer (LeftBound79824.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79823.bound, LeftBound79824.bound]
def bound : CoeffClass := .finite ⟨6056673898188261948960360, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79823.bound, LeftBound79824.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79823.actual selector witness, LeftBound79824.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79825

namespace LeftBound79826
def owner : Owner := ⟨.program ⟨214⟩, ⟨18846⟩⟩
def transferEvent : Nat := 79826
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17662⟩⟩], []⟩ [⟨.result 623 .coefficient, true, some 1⟩, ⟨.result 4385 .coefficient, true, some 1⟩]
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
    BoundInputAt history owner (.result 4385 .coefficient)
      LeftAuthority4384.bound (LeftAuthority4384.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨17662⟩⟩) (rawTerms := some (Proof.Events017.exact4385RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4384.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4384.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority622.bound [LeftAuthority4384.bound]
def bound : CoeffClass := .finite ⟨226487908831958288795280, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority622.bound, LeftAuthority4384.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority622.actual selector witness) * ([LeftAuthority4384.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound79826

namespace LeftBound79827
def owner : Owner := ⟨.program ⟨214⟩, ⟨18846⟩⟩
def transferEvent : Nat := 79827
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 79825, .transfer 79826]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79825)
      LeftBound79825.bound (LeftBound79825.actual selector witness) := by
  exact .transfer (LeftBound79825.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79826)
      LeftBound79826.bound (LeftBound79826.actual selector witness) := by
  exact .transfer (LeftBound79826.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79825.bound, LeftBound79826.bound]
def bound : CoeffClass := .finite ⟨6283161807020220237755640, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79825.bound, LeftBound79826.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79825.actual selector witness, LeftBound79826.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79827

namespace LeftBound79828
def owner : Owner := ⟨.program ⟨214⟩, ⟨18846⟩⟩
def transferEvent : Nat := 79828
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18035⟩⟩], []⟩ [⟨.result 633 .coefficient, true, some 1⟩, ⟨.result 4393 .coefficient, true, some 1⟩]
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
    BoundInputAt history owner (.result 4393 .coefficient)
      LeftAuthority4392.bound (LeftAuthority4392.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨18035⟩⟩) (rawTerms := some (Proof.Events017.exact4393RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4392.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4392.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority632.bound [LeftAuthority4392.bound]
def bound : CoeffClass := .finite ⟨224377773035387248837560, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority632.bound, LeftAuthority4392.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority632.actual selector witness) * ([LeftAuthority4392.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound79828

namespace LeftBound79829
def owner : Owner := ⟨.program ⟨214⟩, ⟨18846⟩⟩
def transferEvent : Nat := 79829
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 79827, .transfer 79828]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79827)
      LeftBound79827.bound (LeftBound79827.actual selector witness) := by
  exact .transfer (LeftBound79827.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79828)
      LeftBound79828.bound (LeftBound79828.actual selector witness) := by
  exact .transfer (LeftBound79828.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79827.bound, LeftBound79828.bound]
def bound : CoeffClass := .finite ⟨6507539580055607486593200, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79827.bound, LeftBound79828.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79827.actual selector witness, LeftBound79828.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79829

namespace LeftBound79830
def owner : Owner := ⟨.program ⟨214⟩, ⟨18846⟩⟩
def transferEvent : Nat := 79830
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17165⟩⟩], []⟩ [⟨.result 643 .coefficient, true, some 1⟩, ⟨.result 4401 .coefficient, true, some 1⟩]
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
    BoundInputAt history owner (.result 4401 .coefficient)
      LeftAuthority4400.bound (LeftAuthority4400.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨17165⟩⟩) (rawTerms := some (Proof.Events017.exact4401RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4400.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4400.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority642.bound [LeftAuthority4400.bound]
def bound : CoeffClass := .finite ⟨222230617312560576599880, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority642.bound, LeftAuthority4400.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority642.actual selector witness) * ([LeftAuthority4400.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound79830

namespace LeftBound79831
def owner : Owner := ⟨.program ⟨214⟩, ⟨18846⟩⟩
def transferEvent : Nat := 79831
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 79829, .transfer 79830]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79829)
      LeftBound79829.bound (LeftBound79829.actual selector witness) := by
  exact .transfer (LeftBound79829.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79830)
      LeftBound79830.bound (LeftBound79830.actual selector witness) := by
  exact .transfer (LeftBound79830.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79829.bound, LeftBound79830.bound]
def bound : CoeffClass := .finite ⟨6729770197368168063193080, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79829.bound, LeftBound79830.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79829.actual selector witness, LeftBound79830.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79831

namespace LeftBound79832
def owner : Owner := ⟨.program ⟨214⟩, ⟨18846⟩⟩
def transferEvent : Nat := 79832
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17221⟩⟩], []⟩ [⟨.result 653 .coefficient, true, some 1⟩, ⟨.result 4409 .coefficient, true, some 1⟩]
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
    BoundInputAt history owner (.result 4409 .coefficient)
      LeftAuthority4408.bound (LeftAuthority4408.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨17221⟩⟩) (rawTerms := some (Proof.Events017.exact4409RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4408.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4408.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority652.bound [LeftAuthority4408.bound]
def bound : CoeffClass := .finite ⟨220778129617707239497920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority652.bound, LeftAuthority4408.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority652.actual selector witness) * ([LeftAuthority4408.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound79832

namespace LeftBound79833
def owner : Owner := ⟨.program ⟨214⟩, ⟨18846⟩⟩
def transferEvent : Nat := 79833
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 79831, .transfer 79832]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79831)
      LeftBound79831.bound (LeftBound79831.actual selector witness) := by
  exact .transfer (LeftBound79831.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79832)
      LeftBound79832.bound (LeftBound79832.actual selector witness) := by
  exact .transfer (LeftBound79832.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79831.bound, LeftBound79832.bound]
def bound : CoeffClass := .finite ⟨6950548326985875302691000, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79831.bound, LeftBound79832.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79831.actual selector witness, LeftBound79832.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79833

namespace LeftBound79834
def owner : Owner := ⟨.program ⟨214⟩, ⟨18846⟩⟩
def transferEvent : Nat := 79834
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17438⟩⟩], []⟩ [⟨.result 663 .coefficient, true, some 1⟩, ⟨.result 4417 .coefficient, true, some 1⟩]
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
    BoundInputAt history owner (.result 4417 .coefficient)
      LeftAuthority4416.bound (LeftAuthority4416.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨17438⟩⟩) (rawTerms := some (Proof.Events017.exact4417RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4416.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4416.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority662.bound [LeftAuthority4416.bound]
def bound : CoeffClass := .finite ⟨216532396355828254122960, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority662.bound, LeftAuthority4416.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority662.actual selector witness) * ([LeftAuthority4416.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound79834

namespace LeftBound79835
def owner : Owner := ⟨.program ⟨214⟩, ⟨18846⟩⟩
def transferEvent : Nat := 79835
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 79833, .transfer 79834]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79833)
      LeftBound79833.bound (LeftBound79833.actual selector witness) := by
  exact .transfer (LeftBound79833.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79834)
      LeftBound79834.bound (LeftBound79834.actual selector witness) := by
  exact .transfer (LeftBound79834.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79833.bound, LeftBound79834.bound]
def bound : CoeffClass := .finite ⟨7167080723341703556813960, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79833.bound, LeftBound79834.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79833.actual selector witness, LeftBound79834.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79835

namespace LeftBound79836
def owner : Owner := ⟨.program ⟨214⟩, ⟨18846⟩⟩
def transferEvent : Nat := 79836
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17814⟩⟩], []⟩ [⟨.result 673 .coefficient, true, some 1⟩, ⟨.result 4425 .coefficient, true, some 1⟩]
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
    BoundInputAt history owner (.result 4425 .coefficient)
      LeftAuthority4424.bound (LeftAuthority4424.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨17814⟩⟩) (rawTerms := some (Proof.Events017.exact4425RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4424.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4424.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority672.bound [LeftAuthority4424.bound]
def bound : CoeffClass := .finite ⟨213251602471649038151400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority672.bound, LeftAuthority4424.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority672.actual selector witness) * ([LeftAuthority4424.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound79836

namespace LeftBound79837
def owner : Owner := ⟨.program ⟨214⟩, ⟨18846⟩⟩
def transferEvent : Nat := 79837
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 79835, .transfer 79836]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79835)
      LeftBound79835.bound (LeftBound79835.actual selector witness) := by
  exact .transfer (LeftBound79835.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79836)
      LeftBound79836.bound (LeftBound79836.actual selector witness) := by
  exact .transfer (LeftBound79836.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79835.bound, LeftBound79836.bound]
def bound : CoeffClass := .finite ⟨7380332325813352594965360, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79835.bound, LeftBound79836.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79835.actual selector witness, LeftBound79836.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79837

namespace LeftBound79838
def owner : Owner := ⟨.program ⟨214⟩, ⟨18846⟩⟩
def transferEvent : Nat := 79838
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15516⟩⟩], []⟩ [⟨.result 683 .coefficient, true, some 1⟩, ⟨.result 4433 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 683 .coefficient)
      LeftAuthority682.bound (LeftAuthority682.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6427⟩⟩) (rawTerms := some (Proof.Events002.exact683RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority682.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority682.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 4433 .coefficient)
      LeftAuthority4432.bound (LeftAuthority4432.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨15516⟩⟩) (rawTerms := some (Proof.Events017.exact4433RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4432.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4432.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority682.bound [LeftAuthority4432.bound]
def bound : CoeffClass := .finite ⟨201065796616126235971320, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority682.bound, LeftAuthority4432.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority682.actual selector witness) * ([LeftAuthority4432.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound79838

namespace LeftBound79839
def owner : Owner := ⟨.program ⟨214⟩, ⟨18846⟩⟩
def transferEvent : Nat := 79839
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 79837, .transfer 79838]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79837)
      LeftBound79837.bound (LeftBound79837.actual selector witness) := by
  exact .transfer (LeftBound79837.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79838)
      LeftBound79838.bound (LeftBound79838.actual selector witness) := by
  exact .transfer (LeftBound79838.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79837.bound, LeftBound79838.bound]
def bound : CoeffClass := .finite ⟨7581398122429478830936680, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79837.bound, LeftBound79838.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79837.actual selector witness, LeftBound79838.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79839

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
