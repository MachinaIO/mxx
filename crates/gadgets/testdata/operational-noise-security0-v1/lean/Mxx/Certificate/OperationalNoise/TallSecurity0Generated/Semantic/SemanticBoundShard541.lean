import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard028
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard538
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard540

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound79840
def owner : Owner := ⟨.program ⟨214⟩, ⟨18846⟩⟩
def transferEvent : Nat := 79840
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15208⟩⟩], []⟩ [⟨.result 693 .coefficient, true, some 1⟩, ⟨.result 4441 .coefficient, true, some 1⟩]
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
    BoundInputAt history owner (.result 4441 .coefficient)
      LeftAuthority4440.bound (LeftAuthority4440.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨15208⟩⟩) (rawTerms := some (Proof.Events017.exact4441RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4440.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4440.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority692.bound [LeftAuthority4440.bound]
def bound : CoeffClass := .finite ⟨187661410175051153573232, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority692.bound, LeftAuthority4440.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority692.actual selector witness) * ([LeftAuthority4440.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound79840

namespace LeftBound79841
def owner : Owner := ⟨.program ⟨214⟩, ⟨18846⟩⟩
def transferEvent : Nat := 79841
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 79839, .transfer 79840]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79839)
      LeftBound79839.bound (LeftBound79839.actual selector witness) := by
  exact .transfer (LeftBound79839.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79840)
      LeftBound79840.bound (LeftBound79840.actual selector witness) := by
  exact .transfer (LeftBound79840.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79839.bound, LeftBound79840.bound]
def bound : CoeffClass := .finite ⟨7769059532604529984509912, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79839.bound, LeftBound79840.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79839.actual selector witness, LeftBound79840.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79841

namespace LeftBound79842
def owner : Owner := ⟨.program ⟨214⟩, ⟨18846⟩⟩
def transferEvent : Nat := 79842
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15047⟩⟩], []⟩ [⟨.result 703 .coefficient, true, some 1⟩, ⟨.result 4449 .coefficient, true, some 1⟩]
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
    BoundInputAt history owner (.result 4449 .coefficient)
      LeftAuthority4448.bound (LeftAuthority4448.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨15047⟩⟩) (rawTerms := some (Proof.Events017.exact4449RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4448.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4448.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority702.bound [LeftAuthority4448.bound]
def bound : CoeffClass := .finite ⟨175932572039110456474905, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority702.bound, LeftAuthority4448.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority702.actual selector witness) * ([LeftAuthority4448.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound79842

namespace LeftBound79843
def owner : Owner := ⟨.program ⟨214⟩, ⟨18846⟩⟩
def transferEvent : Nat := 79843
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 79841, .transfer 79842]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79841)
      LeftBound79841.bound (LeftBound79841.actual selector witness) := by
  exact .transfer (LeftBound79841.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79842)
      LeftBound79842.bound (LeftBound79842.actual selector witness) := by
  exact .transfer (LeftBound79842.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79841.bound, LeftBound79842.bound]
def bound : CoeffClass := .finite ⟨7944992104643640440984817, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79841.bound, LeftBound79842.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79841.actual selector witness, LeftBound79842.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79843

namespace LeftBound79844
def owner : Owner := ⟨.program ⟨214⟩, ⟨18846⟩⟩
def transferEvent : Nat := 79844
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14886⟩⟩], []⟩ [⟨.result 713 .coefficient, true, some 1⟩, ⟨.result 4457 .coefficient, true, some 1⟩]
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
    BoundInputAt history owner (.result 4457 .coefficient)
      LeftAuthority4456.bound (LeftAuthority4456.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨14886⟩⟩) (rawTerms := some (Proof.Events017.exact4457RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4456.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4456.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority712.bound [LeftAuthority4456.bound]
def bound : CoeffClass := .finite ⟨156384508479209294644360, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority712.bound, LeftAuthority4456.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority712.actual selector witness) * ([LeftAuthority4456.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound79844

namespace LeftBound79845
def owner : Owner := ⟨.program ⟨214⟩, ⟨18846⟩⟩
def transferEvent : Nat := 79845
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 79843, .transfer 79844]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79843)
      LeftBound79843.bound (LeftBound79843.actual selector witness) := by
  exact .transfer (LeftBound79843.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79844)
      LeftBound79844.bound (LeftBound79844.actual selector witness) := by
  exact .transfer (LeftBound79844.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79843.bound, LeftBound79844.bound]
def bound : CoeffClass := .finite ⟨8101376613122849735629177, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79843.bound, LeftBound79844.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79843.actual selector witness, LeftBound79844.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79845

namespace LeftBound79846
def owner : Owner := ⟨.program ⟨214⟩, ⟨18846⟩⟩
def transferEvent : Nat := 79846
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 79805 .summary) (.transfer 79845) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79805 .summary)
      LeftBound79803.bound (LeftBound79803.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨7752⟩⟩) (rawTerms := some (Proof.Events311.exact79805RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79803.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79845)
      LeftBound79845.bound (LeftBound79845.actual selector witness) := by
  exact .transfer (LeftBound79845.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound79803.bound LeftBound79845.bound
def bound : CoeffClass := .finite ⟨6740345342118210980043475264, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79803.bound, LeftBound79845.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound79803.actual selector witness) * (LeftBound79845.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound79846

namespace LeftBound79918
def owner : Owner := ⟨.program ⟨214⟩, ⟨6567⟩⟩
def transferEvent : Nat := 79918
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 79916 .coefficient) (.predecessor 1 79917 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79916 .coefficient)
      LeftBound79789.bound (LeftBound79789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events311.exact79790RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79789.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79789.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79917 .coefficient)
      LeftAuthority1.bound (LeftAuthority1.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact2RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound79789.bound LeftAuthority1.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79789.bound, LeftAuthority1.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound79789.actual selector witness) * (LeftAuthority1.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound79918

namespace LeftBound79923
def owner : Owner := ⟨.program ⟨214⟩, ⟨13353⟩⟩
def transferEvent : Nat := 79923
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 79921 .coefficient) (.predecessor 1 79922 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79921 .coefficient)
      LeftAuthority3827.bound (LeftAuthority3827.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events014.exact3828RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3827.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3827.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79922 .coefficient)
      LeftBound79918.bound (LeftBound79918.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79920RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79918.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79918.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority3827.bound LeftBound79918.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3827.bound, LeftBound79918.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority3827.actual selector witness) * (LeftBound79918.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound79923

namespace LeftBound79928
def owner : Owner := ⟨.program ⟨214⟩, ⟨7246⟩⟩
def transferEvent : Nat := 79928
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 79926 .coefficient) (.predecessor 1 79927 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79926 .coefficient)
      LeftBound79789.bound (LeftBound79789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events311.exact79790RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79789.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79789.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79927 .coefficient)
      LeftBound6456.bound (LeftBound6456.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6457RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6456.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6456.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound79789.bound LeftBound6456.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79789.bound, LeftBound6456.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound79789.actual selector witness) * (LeftBound6456.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound79928

namespace LeftBound79933
def owner : Owner := ⟨.program ⟨214⟩, ⟨13354⟩⟩
def transferEvent : Nat := 79933
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 79931 .coefficient, .predecessor 1 79932 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79931 .coefficient)
      LeftBound79928.bound (LeftBound79928.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79930RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79928.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79928.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79932 .coefficient)
      LeftBound79923.bound (LeftBound79923.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79925RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79923.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79923.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79928.bound, LeftBound79923.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79928.bound, LeftBound79923.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79928.actual selector witness, LeftBound79923.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79933

namespace LeftBound79937
def owner : Owner := ⟨.program ⟨214⟩, ⟨13355⟩⟩
def transferEvent : Nat := 79937
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 79935 .coefficient, .predecessor 1 79936 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79935 .coefficient)
      LeftBound79933.bound (LeftBound79933.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79934RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79933.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79933.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79936 .coefficient)
      LeftBound6443.bound (LeftBound6443.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6444RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6443.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6443.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79933.bound, LeftBound6443.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79933.bound, LeftBound6443.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79933.actual selector witness, LeftBound6443.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79937

namespace LeftBound79938
def owner : Owner := ⟨.program ⟨214⟩, ⟨13355⟩⟩
def transferEvent : Nat := 79938
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨104⟩⟩]⟩ [⟨.result 6444 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6444 .coefficient)
      LeftBound6443.bound (LeftBound6443.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨104⟩⟩) (rawTerms := some (Proof.Events025.exact6444RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6443.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6443.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound6443.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6443.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound6443.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound79938

namespace LeftBound79943
def owner : Owner := ⟨.program ⟨214⟩, ⟨13356⟩⟩
def transferEvent : Nat := 79943
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 79941 .coefficient) (.predecessor 1 79942 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79941 .coefficient)
      LeftBound79937.bound (LeftBound79937.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79940RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79937.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79937.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79942 .coefficient)
      LeftAuthority3830.bound (LeftAuthority3830.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events014.exact3831RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3830.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3830.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound79937.bound LeftAuthority3830.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79937.bound, LeftAuthority3830.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound79937.actual selector witness) * (LeftAuthority3830.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound79943

namespace LeftBound79944
def owner : Owner := ⟨.program ⟨214⟩, ⟨13356⟩⟩
def transferEvent : Nat := 79944
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨10345⟩⟩], []⟩ [⟨.result 3831 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 3831 .coefficient)
      LeftAuthority3830.bound (LeftAuthority3830.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨10345⟩⟩) (rawTerms := some (Proof.Events014.exact3831RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3830.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3830.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority3830.bound []
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3830.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority3830.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound79944

namespace LeftBound79945
def owner : Owner := ⟨.program ⟨214⟩, ⟨13356⟩⟩
def transferEvent : Nat := 79945
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 79940 .summary) (.transfer 79944) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79940 .summary)
      LeftBound79938.bound (LeftBound79938.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13355⟩⟩) (rawTerms := some (Proof.Events312.exact79940RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79938.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79944)
      LeftBound79944.bound (LeftBound79944.actual selector witness) := by
  exact .transfer (LeftBound79944.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound79938.bound LeftBound79944.bound
def bound : CoeffClass := .finite ⟨49920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79938.bound, LeftBound79944.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound79938.actual selector witness) * (LeftBound79944.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound79945

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
