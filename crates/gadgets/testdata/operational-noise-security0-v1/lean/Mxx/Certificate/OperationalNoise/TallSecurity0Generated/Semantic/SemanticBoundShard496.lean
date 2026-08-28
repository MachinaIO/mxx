import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard495

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound72620
def owner : Owner := ⟨.program ⟨214⟩, ⟨19167⟩⟩
def transferEvent : Nat := 72620
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 72618 .coefficient) (.predecessor 1 72619 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72618 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72619 .coefficient)
      LeftBound72616.bound (LeftBound72616.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events283.exact72617RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72616.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72616.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound72616.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound72616.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound72616.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72620

namespace LeftBound72621
def owner : Owner := ⟨.program ⟨214⟩, ⟨19167⟩⟩
def transferEvent : Nat := 72621
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19164⟩⟩]⟩ [⟨.result 72613 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 72613 .coefficient)
      LeftAuthority72612.bound (LeftAuthority72612.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19164⟩⟩) (rawTerms := some (Proof.Events283.exact72613RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72612.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72612.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority72612.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority72612.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority72612.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound72621

namespace LeftBound72622
def owner : Owner := ⟨.program ⟨214⟩, ⟨19167⟩⟩
def transferEvent : Nat := 72622
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 72621) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 72621)
      LeftBound72621.bound (LeftBound72621.actual selector witness) := by
  exact .transfer (LeftBound72621.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound72621.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound72621.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound72621.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72622

namespace LeftBound72701
def owner : Owner := ⟨.program ⟨214⟩, ⟨10970⟩⟩
def transferEvent : Nat := 72701
def frameStart : Nat := 72672
def rule : BoundRule := .product (.predecessor 0 72699 .coefficient) (.predecessor 1 72700 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72699 .coefficient)
      LeftAuthority72697.bound (LeftAuthority72697.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events283.exact72698RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72697.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72697.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72700 .coefficient)
      LeftAuthority72694.bound (LeftAuthority72694.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events283.exact72695RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72694.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72694.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority72697.bound LeftAuthority72694.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority72697.bound, LeftAuthority72694.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority72697.actual selector witness) * (LeftAuthority72694.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72701

namespace LeftBound72705
def owner : Owner := ⟨.program ⟨214⟩, ⟨10971⟩⟩
def transferEvent : Nat := 72705
def frameStart : Nat := 72672
def rule : BoundRule := .identity (.predecessor 0 72704 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72704 .coefficient)
      LeftBound72701.bound (LeftBound72701.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events283.exact72703RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72701.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72701.derived selector witness)

def rawBound : CoeffClass := LeftBound72701.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72701.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound72701.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound72705

namespace LeftBound72722
def owner : Owner := ⟨.program ⟨214⟩, ⟨11069⟩⟩
def transferEvent : Nat := 72722
def frameStart : Nat := 72672
def rule : BoundRule := .sum [.predecessor 0 72720 .coefficient, .predecessor 1 72721 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72720 .coefficient)
      LeftBound72705.bound (LeftBound72705.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound72705.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72721 .coefficient)
      LeftAuthority72718.bound (LeftAuthority72718.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority72718.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound72705.bound, LeftAuthority72718.bound]
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72705.bound, LeftAuthority72718.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound72705.actual selector witness, LeftAuthority72718.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72722

namespace LeftBound72725
def owner : Owner := ⟨.program ⟨214⟩, ⟨11070⟩⟩
def transferEvent : Nat := 72725
def frameStart : Nat := 72672
def rule : BoundRule := .identity (.predecessor 0 72724 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72724 .coefficient)
      LeftBound72722.bound (LeftBound72722.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound72722.derived selector witness)

def rawBound : CoeffClass := LeftBound72722.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72722.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound72722.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound72725

namespace LeftBound72731
def owner : Owner := ⟨.program ⟨214⟩, ⟨11071⟩⟩
def transferEvent : Nat := 72731
def frameStart : Nat := 72672
def rule : BoundRule := .product (.predecessor 0 72729 .coefficient) (.predecessor 1 72730 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72729 .coefficient)
      LeftAuthority72727.bound (LeftAuthority72727.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events284.exact72728RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72727.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72727.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72730 .coefficient)
      LeftBound72725.bound (LeftBound72725.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events284.exact72726RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72725.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72725.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority72727.bound LeftBound72725.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority72727.bound, LeftBound72725.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority72727.actual selector witness) * (LeftBound72725.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72731

namespace LeftBound72747
def owner : Owner := ⟨.program ⟨214⟩, ⟨7838⟩⟩
def transferEvent : Nat := 72747
def frameStart : Nat := 72672
def rule : BoundRule := .scale (.predecessor 0 72745 .coefficient) (.value (.predecessor 1 72746 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72745 .coefficient)
      LeftAuthority72743.bound (LeftAuthority72743.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events284.exact72744RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72743.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72743.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72746 .coefficient)
      LeftAuthority72734.bound (LeftAuthority72734.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority72734.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority72743.bound LeftAuthority72734.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority72743.bound, LeftAuthority72734.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority72743.actual selector witness) * (LeftAuthority72734.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound72747

namespace LeftBound72750
def owner : Owner := ⟨.program ⟨214⟩, ⟨6791⟩⟩
def transferEvent : Nat := 72750
def frameStart : Nat := 72672
def rule : BoundRule := .identity (.predecessor 0 72749 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72749 .coefficient)
      LeftAuthority72737.bound (LeftAuthority72737.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events284.exact72738RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72737.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72737.derived selector witness)

def rawBound : CoeffClass := LeftAuthority72737.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority72737.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority72737.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound72750

namespace LeftBound72754
def owner : Owner := ⟨.program ⟨214⟩, ⟨7839⟩⟩
def transferEvent : Nat := 72754
def frameStart : Nat := 72672
def rule : BoundRule := .product (.predecessor 0 72752 .coefficient) (.predecessor 1 72753 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72752 .coefficient)
      LeftBound72750.bound (LeftBound72750.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events284.exact72751RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72750.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72750.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72753 .coefficient)
      LeftBound72747.bound (LeftBound72747.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events284.exact72748RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72747.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72747.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound72750.bound LeftBound72747.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72750.bound, LeftBound72747.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound72750.actual selector witness) * (LeftBound72747.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72754

namespace LeftBound72759
def owner : Owner := ⟨.program ⟨214⟩, ⟨11072⟩⟩
def transferEvent : Nat := 72759
def frameStart : Nat := 72672
def rule : BoundRule := .sum [.predecessor 0 72757 .coefficient, .predecessor 1 72758 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72757 .coefficient)
      LeftBound72754.bound (LeftBound72754.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events284.exact72756RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72754.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72754.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72758 .coefficient)
      LeftBound72731.bound (LeftBound72731.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events284.exact72733RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72731.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72731.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound72754.bound, LeftBound72731.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72754.bound, LeftBound72731.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound72754.actual selector witness, LeftBound72731.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72759

namespace LeftBound72763
def owner : Owner := ⟨.program ⟨214⟩, ⟨25063⟩⟩
def transferEvent : Nat := 72763
def frameStart : Nat := 72672
def rule : BoundRule := .product (.predecessor 0 72761 .coefficient) (.predecessor 1 72762 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72761 .coefficient)
      LeftBound72759.bound (LeftBound72759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events284.exact72760RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72762 .coefficient)
      LeftAuthority72716.bound (LeftAuthority72716.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events284.exact72717RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72716.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72716.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound72759.bound LeftAuthority72716.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72759.bound, LeftAuthority72716.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound72759.actual selector witness) * (LeftAuthority72716.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72763

namespace LeftBound72774
def owner : Owner := ⟨.program ⟨214⟩, ⟨15112⟩⟩
def transferEvent : Nat := 72774
def frameStart : Nat := 72672
def rule : BoundRule := .product (.predecessor 0 72772 .coefficient) (.predecessor 1 72773 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72772 .coefficient)
      LeftAuthority72727.bound (LeftAuthority72727.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events284.exact72728RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72727.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72727.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72773 .coefficient)
      LeftAuthority72770.bound (LeftAuthority72770.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events284.exact72771RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72770.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72770.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority72727.bound LeftAuthority72770.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority72727.bound, LeftAuthority72770.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority72727.actual selector witness) * (LeftAuthority72770.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72774

namespace LeftBound72782
def owner : Owner := ⟨.program ⟨214⟩, ⟨15113⟩⟩
def transferEvent : Nat := 72782
def frameStart : Nat := 72672
def rule : BoundRule := .sum [.predecessor 0 72780 .coefficient, .predecessor 1 72781 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72780 .coefficient)
      LeftAuthority72778.bound (LeftAuthority72778.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events284.exact72779RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72778.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72778.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72781 .coefficient)
      LeftBound72774.bound (LeftBound72774.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events284.exact72776RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72774.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72774.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority72778.bound, LeftBound72774.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority72778.bound, LeftBound72774.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority72778.actual selector witness, LeftBound72774.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72782

namespace LeftBound72786
def owner : Owner := ⟨.program ⟨214⟩, ⟨25064⟩⟩
def transferEvent : Nat := 72786
def frameStart : Nat := 72672
def rule : BoundRule := .sum [.predecessor 0 72784 .coefficient, .predecessor 1 72785 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72784 .coefficient)
      LeftBound72782.bound (LeftBound72782.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events284.exact72783RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72782.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72782.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72785 .coefficient)
      LeftBound72763.bound (LeftBound72763.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events284.exact72768RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72763.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72763.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound72782.bound, LeftBound72763.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72782.bound, LeftBound72763.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound72782.actual selector witness, LeftBound72763.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72786

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
