import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard014
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard538

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound79808
def owner : Owner := ⟨.program ⟨214⟩, ⟨18846⟩⟩
def transferEvent : Nat := 79808
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 79806 .coefficient) (.predecessor 1 79807 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79806 .coefficient)
      LeftBound79802.bound (LeftBound79802.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events311.exact79805RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79802.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79802.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79807 .coefficient)
      LeftBound4537.bound (LeftBound4537.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4538RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4537.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4537.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound79802.bound LeftBound4537.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79802.bound, LeftBound4537.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound79802.actual selector witness) * (LeftBound4537.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound79808

namespace LeftBound79809
def owner : Owner := ⟨.program ⟨214⟩, ⟨18846⟩⟩
def transferEvent : Nat := 79809
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18495⟩⟩], []⟩ [⟨.result 36 .coefficient, true, some 1⟩, ⟨.result 4313 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36 .coefficient)
      LeftAuthority35.bound (LeftAuthority35.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6410⟩⟩) (rawTerms := some (Proof.Events000.exact36RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 4313 .coefficient)
      LeftAuthority4312.bound (LeftAuthority4312.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨18495⟩⟩) (rawTerms := some (Proof.Events016.exact4313RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4312.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4312.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority35.bound [LeftAuthority4312.bound]
def bound : CoeffClass := .finite ⟨4222381728938650955397720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority35.bound, LeftAuthority4312.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority35.actual selector witness) * ([LeftAuthority4312.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound79809

namespace LeftBound79810
def owner : Owner := ⟨.program ⟨214⟩, ⟨18846⟩⟩
def transferEvent : Nat := 79810
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18124⟩⟩], []⟩ [⟨.result 543 .coefficient, true, some 1⟩, ⟨.result 4321 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 543 .coefficient)
      LeftAuthority542.bound (LeftAuthority542.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6435⟩⟩) (rawTerms := some (Proof.Events002.exact543RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority542.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority542.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 4321 .coefficient)
      LeftAuthority4320.bound (LeftAuthority4320.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨18124⟩⟩) (rawTerms := some (Proof.Events016.exact4321RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4320.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4320.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority542.bound [LeftAuthority4320.bound]
def bound : CoeffClass := .finite ⟨230731242018505516688400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority542.bound, LeftAuthority4320.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority542.actual selector witness) * ([LeftAuthority4320.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound79810

namespace LeftBound79811
def owner : Owner := ⟨.program ⟨214⟩, ⟨18846⟩⟩
def transferEvent : Nat := 79811
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 79809, .transfer 79810]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79809)
      LeftBound79809.bound (LeftBound79809.actual selector witness) := by
  exact .transfer (LeftBound79809.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79810)
      LeftBound79810.bound (LeftBound79810.actual selector witness) := by
  exact .transfer (LeftBound79810.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79809.bound, LeftBound79810.bound]
def bound : CoeffClass := .finite ⟨4453112970957156472086120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79809.bound, LeftBound79810.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79809.actual selector witness, LeftBound79810.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79811

namespace LeftBound79812
def owner : Owner := ⟨.program ⟨214⟩, ⟨18846⟩⟩
def transferEvent : Nat := 79812
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16927⟩⟩], []⟩ [⟨.result 553 .coefficient, true, some 1⟩, ⟨.result 4329 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 553 .coefficient)
      LeftAuthority552.bound (LeftAuthority552.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6437⟩⟩) (rawTerms := some (Proof.Events002.exact553RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority552.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority552.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 4329 .coefficient)
      LeftAuthority4328.bound (LeftAuthority4328.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨16927⟩⟩) (rawTerms := some (Proof.Events016.exact4329RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4328.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4328.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority552.bound [LeftAuthority4328.bound]
def bound : CoeffClass := .finite ⟨230600885384596756509480, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority552.bound, LeftAuthority4328.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority552.actual selector witness) * ([LeftAuthority4328.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound79812

namespace LeftBound79813
def owner : Owner := ⟨.program ⟨214⟩, ⟨18846⟩⟩
def transferEvent : Nat := 79813
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 79811, .transfer 79812]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79811)
      LeftBound79811.bound (LeftBound79811.actual selector witness) := by
  exact .transfer (LeftBound79811.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79812)
      LeftBound79812.bound (LeftBound79812.actual selector witness) := by
  exact .transfer (LeftBound79812.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79811.bound, LeftBound79812.bound]
def bound : CoeffClass := .finite ⟨4683713856341753228595600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79811.bound, LeftBound79812.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79811.actual selector witness, LeftBound79812.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79813

namespace LeftBound79814
def owner : Owner := ⟨.program ⟨214⟩, ⟨18846⟩⟩
def transferEvent : Nat := 79814
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17494⟩⟩], []⟩ [⟨.result 563 .coefficient, true, some 1⟩, ⟨.result 4337 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 563 .coefficient)
      LeftAuthority562.bound (LeftAuthority562.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6449⟩⟩) (rawTerms := some (Proof.Events002.exact563RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority562.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority562.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 4337 .coefficient)
      LeftAuthority4336.bound (LeftAuthority4336.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨17494⟩⟩) (rawTerms := some (Proof.Events016.exact4337RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4336.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4336.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority562.bound [LeftAuthority4336.bound]
def bound : CoeffClass := .finite ⟨230150786063741980797360, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority562.bound, LeftAuthority4336.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority562.actual selector witness) * ([LeftAuthority4336.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound79814

namespace LeftBound79815
def owner : Owner := ⟨.program ⟨214⟩, ⟨18846⟩⟩
def transferEvent : Nat := 79815
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 79813, .transfer 79814]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79813)
      LeftBound79813.bound (LeftBound79813.actual selector witness) := by
  exact .transfer (LeftBound79813.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79814)
      LeftBound79814.bound (LeftBound79814.actual selector witness) := by
  exact .transfer (LeftBound79814.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79813.bound, LeftBound79814.bound]
def bound : CoeffClass := .finite ⟨4913864642405495209392960, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79813.bound, LeftBound79814.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79813.actual selector witness, LeftBound79814.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79815

namespace LeftBound79816
def owner : Owner := ⟨.program ⟨214⟩, ⟨18846⟩⟩
def transferEvent : Nat := 79816
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17718⟩⟩], []⟩ [⟨.result 573 .coefficient, true, some 1⟩, ⟨.result 4345 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 573 .coefficient)
      LeftAuthority572.bound (LeftAuthority572.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6459⟩⟩) (rawTerms := some (Proof.Events002.exact573RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority572.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority572.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 4345 .coefficient)
      LeftAuthority4344.bound (LeftAuthority4344.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨17718⟩⟩) (rawTerms := some (Proof.Events016.exact4345RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4344.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4344.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority572.bound [LeftAuthority4344.bound]
def bound : CoeffClass := .finite ⟨229585767767349815541720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority572.bound, LeftAuthority4344.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority572.actual selector witness) * ([LeftAuthority4344.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound79816

namespace LeftBound79817
def owner : Owner := ⟨.program ⟨214⟩, ⟨18846⟩⟩
def transferEvent : Nat := 79817
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 79815, .transfer 79816]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79815)
      LeftBound79815.bound (LeftBound79815.actual selector witness) := by
  exact .transfer (LeftBound79815.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79816)
      LeftBound79816.bound (LeftBound79816.actual selector witness) := by
  exact .transfer (LeftBound79816.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79815.bound, LeftBound79816.bound]
def bound : CoeffClass := .finite ⟨5143450410172845024934680, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79815.bound, LeftBound79816.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79815.actual selector witness, LeftBound79816.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79817

namespace LeftBound79818
def owner : Owner := ⟨.program ⟨214⟩, ⟨18846⟩⟩
def transferEvent : Nat := 79818
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17949⟩⟩], []⟩ [⟨.result 583 .coefficient, true, some 1⟩, ⟨.result 4353 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 583 .coefficient)
      LeftAuthority582.bound (LeftAuthority582.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6467⟩⟩) (rawTerms := some (Proof.Events002.exact583RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority582.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority582.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 4353 .coefficient)
      LeftAuthority4352.bound (LeftAuthority4352.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨17949⟩⟩) (rawTerms := some (Proof.Events017.exact4353RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4352.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4352.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority582.bound [LeftAuthority4352.bound]
def bound : CoeffClass := .finite ⟨229121489167213617734760, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority582.bound, LeftAuthority4352.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority582.actual selector witness) * ([LeftAuthority4352.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound79818

namespace LeftBound79819
def owner : Owner := ⟨.program ⟨214⟩, ⟨18846⟩⟩
def transferEvent : Nat := 79819
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 79817, .transfer 79818]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79817)
      LeftBound79817.bound (LeftBound79817.actual selector witness) := by
  exact .transfer (LeftBound79817.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79818)
      LeftBound79818.bound (LeftBound79818.actual selector witness) := by
  exact .transfer (LeftBound79818.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79817.bound, LeftBound79818.bound]
def bound : CoeffClass := .finite ⟨5372571899340058642669440, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79817.bound, LeftBound79818.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79817.actual selector witness, LeftBound79818.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79819

namespace LeftBound79820
def owner : Owner := ⟨.program ⟨214⟩, ⟨18846⟩⟩
def transferEvent : Nat := 79820
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17550⟩⟩], []⟩ [⟨.result 593 .coefficient, true, some 1⟩, ⟨.result 4361 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 593 .coefficient)
      LeftAuthority592.bound (LeftAuthority592.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6473⟩⟩) (rawTerms := some (Proof.Events002.exact593RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority592.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority592.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 4361 .coefficient)
      LeftAuthority4360.bound (LeftAuthority4360.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨17550⟩⟩) (rawTerms := some (Proof.Events017.exact4361RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4360.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4360.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority592.bound [LeftAuthority4360.bound]
def bound : CoeffClass := .finite ⟨228855378262257504357600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority592.bound, LeftAuthority4360.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority592.actual selector witness) * ([LeftAuthority4360.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound79820

namespace LeftBound79821
def owner : Owner := ⟨.program ⟨214⟩, ⟨18846⟩⟩
def transferEvent : Nat := 79821
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 79819, .transfer 79820]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79819)
      LeftBound79819.bound (LeftBound79819.actual selector witness) := by
  exact .transfer (LeftBound79819.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79820)
      LeftBound79820.bound (LeftBound79820.actual selector witness) := by
  exact .transfer (LeftBound79820.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79819.bound, LeftBound79820.bound]
def bound : CoeffClass := .finite ⟨5601427277602316147027040, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79819.bound, LeftBound79820.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79819.actual selector witness, LeftBound79820.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79821

namespace LeftBound79822
def owner : Owner := ⟨.program ⟨214⟩, ⟨18846⟩⟩
def transferEvent : Nat := 79822
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18832⟩⟩], []⟩ [⟨.result 603 .coefficient, true, some 1⟩, ⟨.result 4369 .coefficient, true, some 1⟩]
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
    BoundInputAt history owner (.result 4369 .coefficient)
      LeftAuthority4368.bound (LeftAuthority4368.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨18832⟩⟩) (rawTerms := some (Proof.Events017.exact4369RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4368.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4368.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority602.bound [LeftAuthority4368.bound]
def bound : CoeffClass := .finite ⟨228236850212900051643120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority602.bound, LeftAuthority4368.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority602.actual selector witness) * ([LeftAuthority4368.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound79822

namespace LeftBound79823
def owner : Owner := ⟨.program ⟨214⟩, ⟨18846⟩⟩
def transferEvent : Nat := 79823
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 79821, .transfer 79822]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79821)
      LeftBound79821.bound (LeftBound79821.actual selector witness) := by
  exact .transfer (LeftBound79821.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79822)
      LeftBound79822.bound (LeftBound79822.actual selector witness) := by
  exact .transfer (LeftBound79822.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79821.bound, LeftBound79822.bound]
def bound : CoeffClass := .finite ⟨5829664127815216198670160, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79821.bound, LeftBound79822.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79821.actual selector witness, LeftBound79822.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79823

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
