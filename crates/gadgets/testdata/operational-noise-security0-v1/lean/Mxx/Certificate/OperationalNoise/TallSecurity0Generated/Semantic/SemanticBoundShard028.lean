import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard025
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard026
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard027

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound6362
def owner : Owner := ⟨.program ⟨214⟩, ⟨18906⟩⟩
def transferEvent : Nat := 6362
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15536⟩⟩], []⟩ [⟨.result 683 .coefficient, true, some 1⟩, ⟨.result 686 .coefficient, true, some 1⟩]
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
    BoundInputAt history owner (.result 686 .coefficient)
      LeftAuthority685.bound (LeftAuthority685.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨15536⟩⟩) (rawTerms := some (Proof.Events002.exact686RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority685.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority685.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority682.bound [LeftAuthority685.bound]
def bound : CoeffClass := .finite ⟨201065796616126235971320, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority682.bound, LeftAuthority685.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority682.actual selector witness) * ([LeftAuthority685.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound6362

namespace LeftBound6363
def owner : Owner := ⟨.program ⟨214⟩, ⟨18906⟩⟩
def transferEvent : Nat := 6363
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 6361, .transfer 6362]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 6361)
      LeftBound6361.bound (LeftBound6361.actual selector witness) := by
  exact .transfer (LeftBound6361.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 6362)
      LeftBound6362.bound (LeftBound6362.actual selector witness) := by
  exact .transfer (LeftBound6362.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6361.bound, LeftBound6362.bound]
def bound : CoeffClass := .finite ⟨7581398122429478830936680, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6361.bound, LeftBound6362.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound6361.actual selector witness, LeftBound6362.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6363

namespace LeftBound6364
def owner : Owner := ⟨.program ⟨214⟩, ⟨18906⟩⟩
def transferEvent : Nat := 6364
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15228⟩⟩], []⟩ [⟨.result 693 .coefficient, true, some 1⟩, ⟨.result 696 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 693 .coefficient)
      LeftAuthority692.bound (LeftAuthority692.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6452⟩⟩) (rawTerms := some (Proof.Events002.exact693RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority692.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority692.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 696 .coefficient)
      LeftAuthority695.bound (LeftAuthority695.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨15228⟩⟩) (rawTerms := some (Proof.Events002.exact696RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority695.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority695.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority692.bound [LeftAuthority695.bound]
def bound : CoeffClass := .finite ⟨187661410175051153573232, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority692.bound, LeftAuthority695.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority692.actual selector witness) * ([LeftAuthority695.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound6364

namespace LeftBound6365
def owner : Owner := ⟨.program ⟨214⟩, ⟨18906⟩⟩
def transferEvent : Nat := 6365
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 6363, .transfer 6364]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 6363)
      LeftBound6363.bound (LeftBound6363.actual selector witness) := by
  exact .transfer (LeftBound6363.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 6364)
      LeftBound6364.bound (LeftBound6364.actual selector witness) := by
  exact .transfer (LeftBound6364.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6363.bound, LeftBound6364.bound]
def bound : CoeffClass := .finite ⟨7769059532604529984509912, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6363.bound, LeftBound6364.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound6363.actual selector witness, LeftBound6364.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6365

namespace LeftBound6366
def owner : Owner := ⟨.program ⟨214⟩, ⟨18906⟩⟩
def transferEvent : Nat := 6366
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], []⟩ [⟨.result 703 .coefficient, true, some 1⟩, ⟨.result 706 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 703 .coefficient)
      LeftAuthority702.bound (LeftAuthority702.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6475⟩⟩) (rawTerms := some (Proof.Events002.exact703RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority702.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority702.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 706 .coefficient)
      LeftAuthority705.bound (LeftAuthority705.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨15067⟩⟩) (rawTerms := some (Proof.Events002.exact706RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority705.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority705.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority702.bound [LeftAuthority705.bound]
def bound : CoeffClass := .finite ⟨175932572039110456474905, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority702.bound, LeftAuthority705.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority702.actual selector witness) * ([LeftAuthority705.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound6366

namespace LeftBound6367
def owner : Owner := ⟨.program ⟨214⟩, ⟨18906⟩⟩
def transferEvent : Nat := 6367
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 6365, .transfer 6366]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 6365)
      LeftBound6365.bound (LeftBound6365.actual selector witness) := by
  exact .transfer (LeftBound6365.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 6366)
      LeftBound6366.bound (LeftBound6366.actual selector witness) := by
  exact .transfer (LeftBound6366.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6365.bound, LeftBound6366.bound]
def bound : CoeffClass := .finite ⟨7944992104643640440984817, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6365.bound, LeftBound6366.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound6365.actual selector witness, LeftBound6366.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6367

namespace LeftBound6368
def owner : Owner := ⟨.program ⟨214⟩, ⟨18906⟩⟩
def transferEvent : Nat := 6368
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], []⟩ [⟨.result 713 .coefficient, true, some 1⟩, ⟨.result 716 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 713 .coefficient)
      LeftAuthority712.bound (LeftAuthority712.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6495⟩⟩) (rawTerms := some (Proof.Events002.exact713RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority712.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority712.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 716 .coefficient)
      LeftAuthority715.bound (LeftAuthority715.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨14906⟩⟩) (rawTerms := some (Proof.Events002.exact716RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority715.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority715.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority712.bound [LeftAuthority715.bound]
def bound : CoeffClass := .finite ⟨156384508479209294644360, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority712.bound, LeftAuthority715.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority712.actual selector witness) * ([LeftAuthority715.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound6368

namespace LeftBound6369
def owner : Owner := ⟨.program ⟨214⟩, ⟨18906⟩⟩
def transferEvent : Nat := 6369
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 6367, .transfer 6368]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 6367)
      LeftBound6367.bound (LeftBound6367.actual selector witness) := by
  exact .transfer (LeftBound6367.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 6368)
      LeftBound6368.bound (LeftBound6368.actual selector witness) := by
  exact .transfer (LeftBound6368.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6367.bound, LeftBound6368.bound]
def bound : CoeffClass := .finite ⟨8101376613122849735629177, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6367.bound, LeftBound6368.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound6367.actual selector witness, LeftBound6368.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6369

namespace LeftBound6370
def owner : Owner := ⟨.program ⟨214⟩, ⟨18906⟩⟩
def transferEvent : Nat := 6370
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6329 .summary) (.transfer 6369) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6329 .summary)
      LeftBound6327.bound (LeftBound6327.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨7768⟩⟩) (rawTerms := some (Proof.Events024.exact6329RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6327.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 6369)
      LeftBound6369.bound (LeftBound6369.actual selector witness) := by
  exact .transfer (LeftBound6369.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound6327.bound LeftBound6369.bound
def bound : CoeffClass := .finite ⟨6740345342118210980043475264, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6327.bound, LeftBound6369.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound6327.actual selector witness) * (LeftBound6369.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound6370

namespace LeftBound6443
def owner : Owner := ⟨.program ⟨214⟩, ⟨104⟩⟩
def transferEvent : Nat := 6443
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 6442 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6442 .coefficient)
      LeftAuthority6440.bound (LeftAuthority6440.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6441RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6440.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6440.derived selector witness)

def rawBound : CoeffClass := LeftAuthority6440.bound
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6440.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority6440.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound6443

namespace LeftBound6447
def owner : Owner := ⟨.program ⟨214⟩, ⟨6571⟩⟩
def transferEvent : Nat := 6447
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 6445 .coefficient) (.predecessor 1 6446 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6445 .coefficient)
      LeftBound6313.bound (LeftBound6313.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events024.exact6314RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6313.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6313.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6446 .coefficient)
      LeftAuthority1.bound (LeftAuthority1.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact2RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound6313.bound LeftAuthority1.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6313.bound, LeftAuthority1.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound6313.actual selector witness) * (LeftAuthority1.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound6447

namespace LeftBound6452
def owner : Owner := ⟨.program ⟨214⟩, ⟨13385⟩⟩
def transferEvent : Nat := 6452
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 6450 .coefficient) (.predecessor 1 6451 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6450 .coefficient)
      LeftAuthority50.bound (LeftAuthority50.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact51RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority50.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority50.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6451 .coefficient)
      LeftBound6447.bound (LeftBound6447.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6449RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6447.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6447.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority50.bound LeftBound6447.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority50.bound, LeftBound6447.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority50.actual selector witness) * (LeftBound6447.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound6452

namespace LeftBound6456
def owner : Owner := ⟨.program ⟨214⟩, ⟨6790⟩⟩
def transferEvent : Nat := 6456
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 6455 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6455 .coefficient)
      LeftAuthority5869.bound (LeftAuthority5869.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5870RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5869.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5869.derived selector witness)

def rawBound : CoeffClass := LeftAuthority5869.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5869.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority5869.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound6456

namespace LeftBound6460
def owner : Owner := ⟨.program ⟨214⟩, ⟨7398⟩⟩
def transferEvent : Nat := 6460
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 6458 .coefficient) (.predecessor 1 6459 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6458 .coefficient)
      LeftBound6313.bound (LeftBound6313.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events024.exact6314RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6313.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6313.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6459 .coefficient)
      LeftBound6456.bound (LeftBound6456.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6457RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6456.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6456.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound6313.bound LeftBound6456.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6313.bound, LeftBound6456.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound6313.actual selector witness) * (LeftBound6456.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound6460

namespace LeftBound6465
def owner : Owner := ⟨.program ⟨214⟩, ⟨13386⟩⟩
def transferEvent : Nat := 6465
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 6463 .coefficient, .predecessor 1 6464 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6463 .coefficient)
      LeftBound6460.bound (LeftBound6460.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6460.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6464 .coefficient)
      LeftBound6452.bound (LeftBound6452.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6454RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6452.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6452.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6460.bound, LeftBound6452.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6460.bound, LeftBound6452.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound6460.actual selector witness, LeftBound6452.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6465

namespace LeftBound6469
def owner : Owner := ⟨.program ⟨214⟩, ⟨13387⟩⟩
def transferEvent : Nat := 6469
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 6467 .coefficient, .predecessor 1 6468 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6467 .coefficient)
      LeftBound6465.bound (LeftBound6465.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6466RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6465.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6465.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6468 .coefficient)
      LeftBound6443.bound (LeftBound6443.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6444RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6443.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6443.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6465.bound, LeftBound6443.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6465.bound, LeftBound6443.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound6465.actual selector witness, LeftBound6443.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6469

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
