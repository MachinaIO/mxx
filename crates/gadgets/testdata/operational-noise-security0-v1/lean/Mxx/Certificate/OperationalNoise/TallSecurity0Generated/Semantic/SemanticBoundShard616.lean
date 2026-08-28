import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard018
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard554
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard555
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard615

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound90786
def owner : Owner := ⟨.program ⟨214⟩, ⟨17497⟩⟩
def transferEvent : Nat := 90786
def frameStart : Nat := 90690
def rule : BoundRule := .sum [.predecessor 0 90784 .coefficient, .predecessor 1 90785 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90784 .coefficient)
      LeftAuthority90782.bound (LeftAuthority90782.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events354.exact90783RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90782.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90782.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 90785 .coefficient)
      LeftBound90778.bound (LeftBound90778.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events354.exact90780RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90778.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90778.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority90782.bound, LeftBound90778.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority90782.bound, LeftBound90778.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority90782.actual selector witness, LeftBound90778.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90786

namespace LeftBound90790
def owner : Owner := ⟨.program ⟨214⟩, ⟨29601⟩⟩
def transferEvent : Nat := 90790
def frameStart : Nat := 90690
def rule : BoundRule := .sum [.predecessor 0 90788 .coefficient, .predecessor 1 90789 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90788 .coefficient)
      LeftBound90786.bound (LeftBound90786.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events354.exact90787RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90786.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90786.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 90789 .coefficient)
      LeftBound90767.bound (LeftBound90767.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events354.exact90772RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90767.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90767.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound90786.bound, LeftBound90767.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90786.bound, LeftBound90767.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound90786.actual selector witness, LeftBound90767.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90790

namespace LeftBound90803
def owner : Owner := ⟨.program ⟨214⟩, ⟨29598⟩⟩
def transferEvent : Nat := 90803
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 90801 .coefficient, .predecessor 1 90802 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90801 .coefficient)
      LeftBound90632.bound (LeftBound90632.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events354.exact90800RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90632.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90632.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 90802 .coefficient)
      LeftBound90615.bound (LeftBound90615.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90622RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90615.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90615.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound90632.bound, LeftBound90615.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90632.bound, LeftBound90615.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound90632.actual selector witness, LeftBound90615.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90803

namespace LeftBound90806
def owner : Owner := ⟨.program ⟨214⟩, ⟨29598⟩⟩
def transferEvent : Nat := 90806
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 90800 .summary, .result 90622 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 90800 .summary)
      LeftBound90634.bound (LeftBound90634.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22483⟩⟩) (rawTerms := some (Proof.Events354.exact90800RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound90634.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 90622 .summary)
      LeftBound90617.bound (LeftBound90617.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29597⟩⟩) (rawTerms := some (Proof.Events353.exact90622RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound90617.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound90634.bound, LeftBound90617.bound]
def bound : CoeffClass := .finite ⟨1292449485504936292352, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90634.bound, LeftBound90617.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound90634.actual selector witness, LeftBound90617.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90806

namespace LeftBound90810
def owner : Owner := ⟨.program ⟨214⟩, ⟨29599⟩⟩
def transferEvent : Nat := 90810
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 90808 .coefficient) (.predecessor 1 90809 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90808 .coefficient)
      LeftBound90803.bound (LeftBound90803.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events354.exact90807RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90803.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90803.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 90809 .coefficient)
      LeftBound5558.bound (LeftBound5558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5559RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5558.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound90803.bound LeftBound5558.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90803.bound, LeftBound5558.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound90803.actual selector witness) * (LeftBound5558.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90810

namespace LeftBound90811
def owner : Owner := ⟨.program ⟨214⟩, ⟨29599⟩⟩
def transferEvent : Nat := 90811
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6661⟩⟩]⟩ [⟨.result 5555 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5555 .coefficient)
      LeftAuthority5554.bound (LeftAuthority5554.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6661⟩⟩) (rawTerms := some (Proof.Events021.exact5555RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5554.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5554.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5554.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5554.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5554.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound90811

namespace LeftBound90812
def owner : Owner := ⟨.program ⟨214⟩, ⟨29599⟩⟩
def transferEvent : Nat := 90812
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 90807 .summary) (.transfer 90811) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 90807 .summary)
      LeftBound90806.bound (LeftBound90806.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29598⟩⟩) (rawTerms := some (Proof.Events354.exact90807RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound90806.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 90811)
      LeftBound90811.bound (LeftBound90811.actual selector witness) := by
  exact .transfer (LeftBound90811.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound90806.bound LeftBound90811.bound
def bound : CoeffClass := .finite ⟨4743310290994884271912517632, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90806.bound, LeftBound90811.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound90806.actual selector witness) * (LeftBound90811.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90812

namespace LeftBound90827
def owner : Owner := ⟨.program ⟨214⟩, ⟨29380⟩⟩
def transferEvent : Nat := 90827
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 90825 .coefficient) (.predecessor 1 90826 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90825 .coefficient)
      LeftBound81632.bound (LeftBound81632.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events318.exact81636RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81632.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81632.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 90826 .coefficient)
      LeftAuthority90823.bound (LeftAuthority90823.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events354.exact90824RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90823.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90823.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound81632.bound LeftAuthority90823.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81632.bound, LeftAuthority90823.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound81632.actual selector witness) * (LeftAuthority90823.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90827

namespace LeftBound90828
def owner : Owner := ⟨.program ⟨214⟩, ⟨29380⟩⟩
def transferEvent : Nat := 90828
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29378⟩⟩]⟩ [⟨.result 90824 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 90824 .coefficient)
      LeftAuthority90823.bound (LeftAuthority90823.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29378⟩⟩) (rawTerms := some (Proof.Events354.exact90824RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90823.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90823.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority90823.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority90823.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority90823.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound90828

namespace LeftBound90829
def owner : Owner := ⟨.program ⟨214⟩, ⟨29380⟩⟩
def transferEvent : Nat := 90829
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 81636 .summary) (.transfer 90828) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 81636 .summary)
      LeftBound81635.bound (LeftBound81635.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25529⟩⟩) (rawTerms := some (Proof.Events318.exact81636RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound81635.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 90828)
      LeftBound90828.bound (LeftBound90828.actual selector witness) := by
  exact .transfer (LeftBound90828.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound81635.bound LeftBound90828.bound
def bound : CoeffClass := .finite ⟨1292382246358571024384, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81635.bound, LeftBound90828.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound81635.actual selector witness) * (LeftBound90828.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90829

namespace LeftBound90840
def owner : Owner := ⟨.program ⟨214⟩, ⟨22338⟩⟩
def transferEvent : Nat := 90840
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 90838 .coefficient) (.value (.predecessor 1 90839 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90838 .coefficient)
      LeftAuthority90836.bound (LeftAuthority90836.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events354.exact90837RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90836.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90836.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 90839 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority90836.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority90836.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority90836.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound90840

namespace LeftBound90844
def owner : Owner := ⟨.program ⟨214⟩, ⟨22339⟩⟩
def transferEvent : Nat := 90844
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 90842 .coefficient) (.predecessor 1 90843 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90842 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 90843 .coefficient)
      LeftBound90840.bound (LeftBound90840.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events354.exact90841RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90840.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90840.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound90840.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound90840.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound90840.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90844

namespace LeftBound90845
def owner : Owner := ⟨.program ⟨214⟩, ⟨22339⟩⟩
def transferEvent : Nat := 90845
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22336⟩⟩]⟩ [⟨.result 90837 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 90837 .coefficient)
      LeftAuthority90836.bound (LeftAuthority90836.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22336⟩⟩) (rawTerms := some (Proof.Events354.exact90837RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90836.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90836.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority90836.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority90836.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority90836.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound90845

namespace LeftBound90846
def owner : Owner := ⟨.program ⟨214⟩, ⟨22339⟩⟩
def transferEvent : Nat := 90846
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 90845) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 90845)
      LeftBound90845.bound (LeftBound90845.actual selector witness) := by
  exact .transfer (LeftBound90845.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound90845.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound90845.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound90845.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90846

namespace LeftBound90941
def owner : Owner := ⟨.program ⟨214⟩, ⟨16634⟩⟩
def transferEvent : Nat := 90941
def frameStart : Nat := 90902
def rule : BoundRule := .identity (.predecessor 0 90940 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90940 .coefficient)
      LeftAuthority90938.bound (LeftAuthority90938.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact90939RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90938.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90938.derived selector witness)

def rawBound : CoeffClass := LeftAuthority90938.bound
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority90938.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority90938.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound90941

namespace LeftBound90958
def owner : Owner := ⟨.program ⟨214⟩, ⟨16708⟩⟩
def transferEvent : Nat := 90958
def frameStart : Nat := 90902
def rule : BoundRule := .sum [.predecessor 0 90956 .coefficient, .predecessor 1 90957 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90956 .coefficient)
      LeftBound90941.bound (LeftBound90941.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound90941.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 90957 .coefficient)
      LeftAuthority90954.bound (LeftAuthority90954.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority90954.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound90941.bound, LeftAuthority90954.bound]
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90941.bound, LeftAuthority90954.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound90941.actual selector witness, LeftAuthority90954.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90958

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
