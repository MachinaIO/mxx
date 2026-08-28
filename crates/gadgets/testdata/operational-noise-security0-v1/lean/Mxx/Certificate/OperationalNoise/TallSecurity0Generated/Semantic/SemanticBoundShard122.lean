import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard082
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard121

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound19831
def owner : Owner := ⟨.program ⟨214⟩, ⟨27480⟩⟩
def transferEvent : Nat := 19831
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 19829 .coefficient, .predecessor 1 19830 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19829 .coefficient)
      LeftBound19660.bound (LeftBound19660.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events077.exact19828RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19660.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19660.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19830 .coefficient)
      LeftBound19643.bound (LeftBound19643.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events076.exact19650RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19643.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19643.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound19660.bound, LeftBound19643.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19660.bound, LeftBound19643.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound19660.actual selector witness, LeftBound19643.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound19831

namespace LeftBound19834
def owner : Owner := ⟨.program ⟨214⟩, ⟨27480⟩⟩
def transferEvent : Nat := 19834
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 19828 .summary, .result 19650 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 19828 .summary)
      LeftBound19662.bound (LeftBound19662.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21059⟩⟩) (rawTerms := some (Proof.Events077.exact19828RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound19662.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 19650 .summary)
      LeftBound19645.bound (LeftBound19645.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27479⟩⟩) (rawTerms := some (Proof.Events076.exact19650RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound19645.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound19662.bound, LeftBound19645.bound]
def bound : CoeffClass := .finite ⟨1292001236604524572672, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19662.bound, LeftBound19645.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound19662.actual selector witness, LeftBound19645.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound19834

namespace LeftBound19838
def owner : Owner := ⟨.program ⟨214⟩, ⟨27481⟩⟩
def transferEvent : Nat := 19838
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 19836 .coefficient) (.predecessor 1 19837 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19836 .coefficient)
      LeftBound19831.bound (LeftBound19831.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events077.exact19835RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19831.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19831.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19837 .coefficient)
      LeftBound5758.bound (LeftBound5758.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5759RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5758.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5758.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound19831.bound LeftBound5758.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19831.bound, LeftBound5758.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound19831.actual selector witness) * (LeftBound5758.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound19838

namespace LeftBound19839
def owner : Owner := ⟨.program ⟨214⟩, ⟨27481⟩⟩
def transferEvent : Nat := 19839
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩ [⟨.result 5755 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5755 .coefficient)
      LeftAuthority5754.bound (LeftAuthority5754.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6647⟩⟩) (rawTerms := some (Proof.Events022.exact5755RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5754.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5754.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5754.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5754.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5754.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound19839

namespace LeftBound19840
def owner : Owner := ⟨.program ⟨214⟩, ⟨27481⟩⟩
def transferEvent : Nat := 19840
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 19835 .summary) (.transfer 19839) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 19835 .summary)
      LeftBound19834.bound (LeftBound19834.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27480⟩⟩) (rawTerms := some (Proof.Events077.exact19835RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound19834.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 19839)
      LeftBound19839.bound (LeftBound19839.actual selector witness) := by
  exact .transfer (LeftBound19839.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound19834.bound LeftBound19839.bound
def bound : CoeffClass := .finite ⟨4741665210358390854099402752, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19834.bound, LeftBound19839.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound19834.actual selector witness) * (LeftBound19839.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound19840

namespace LeftBound19855
def owner : Owner := ⟨.program ⟨214⟩, ⟨27262⟩⟩
def transferEvent : Nat := 19855
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 19853 .coefficient) (.predecessor 1 19854 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19853 .coefficient)
      LeftBound13256.bound (LeftBound13256.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events051.exact13260RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13256.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13256.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19854 .coefficient)
      LeftAuthority19851.bound (LeftAuthority19851.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events077.exact19852RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19851.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19851.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound13256.bound LeftAuthority19851.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13256.bound, LeftAuthority19851.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound13256.actual selector witness) * (LeftAuthority19851.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound19855

namespace LeftBound19856
def owner : Owner := ⟨.program ⟨214⟩, ⟨27262⟩⟩
def transferEvent : Nat := 19856
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27260⟩⟩]⟩ [⟨.result 19852 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 19852 .coefficient)
      LeftAuthority19851.bound (LeftAuthority19851.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27260⟩⟩) (rawTerms := some (Proof.Events077.exact19852RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19851.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19851.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority19851.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19851.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority19851.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound19856

namespace LeftBound19857
def owner : Owner := ⟨.program ⟨214⟩, ⟨27262⟩⟩
def transferEvent : Nat := 19857
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 13260 .summary) (.transfer 19856) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13260 .summary)
      LeftBound13259.bound (LeftBound13259.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25857⟩⟩) (rawTerms := some (Proof.Events051.exact13260RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound13259.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 19856)
      LeftBound19856.bound (LeftBound19856.actual selector witness) := by
  exact .transfer (LeftBound19856.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound13259.bound LeftBound19856.bound
def bound : CoeffClass := .finite ⟨1291978822348200476672, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13259.bound, LeftBound19856.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound13259.actual selector witness) * (LeftBound19856.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound19857

namespace LeftBound19868
def owner : Owner := ⟨.program ⟨214⟩, ⟨20914⟩⟩
def transferEvent : Nat := 19868
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 19866 .coefficient) (.value (.predecessor 1 19867 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19866 .coefficient)
      LeftAuthority19864.bound (LeftAuthority19864.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events077.exact19865RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19864.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19864.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19867 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority19864.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19864.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority19864.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound19868

namespace LeftBound19872
def owner : Owner := ⟨.program ⟨214⟩, ⟨20915⟩⟩
def transferEvent : Nat := 19872
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 19870 .coefficient) (.predecessor 1 19871 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19870 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19871 .coefficient)
      LeftBound19868.bound (LeftBound19868.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events077.exact19869RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19868.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19868.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound19868.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound19868.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound19868.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound19872

namespace LeftBound19873
def owner : Owner := ⟨.program ⟨214⟩, ⟨20915⟩⟩
def transferEvent : Nat := 19873
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20912⟩⟩]⟩ [⟨.result 19865 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 19865 .coefficient)
      LeftAuthority19864.bound (LeftAuthority19864.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20912⟩⟩) (rawTerms := some (Proof.Events077.exact19865RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19864.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19864.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority19864.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19864.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority19864.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound19873

namespace LeftBound19874
def owner : Owner := ⟨.program ⟨214⟩, ⟨20915⟩⟩
def transferEvent : Nat := 19874
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 19873) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 19873)
      LeftBound19873.bound (LeftBound19873.actual selector witness) := by
  exact .transfer (LeftBound19873.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound19873.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound19873.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound19873.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound19874

namespace LeftBound19969
def owner : Owner := ⟨.program ⟨214⟩, ⟨15600⟩⟩
def transferEvent : Nat := 19969
def frameStart : Nat := 19930
def rule : BoundRule := .identity (.predecessor 0 19968 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19968 .coefficient)
      LeftAuthority19966.bound (LeftAuthority19966.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events077.exact19967RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19966.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19966.derived selector witness)

def rawBound : CoeffClass := LeftAuthority19966.bound
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19966.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority19966.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound19969

namespace LeftBound19986
def owner : Owner := ⟨.program ⟨214⟩, ⟨15674⟩⟩
def transferEvent : Nat := 19986
def frameStart : Nat := 19930
def rule : BoundRule := .sum [.predecessor 0 19984 .coefficient, .predecessor 1 19985 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19984 .coefficient)
      LeftBound19969.bound (LeftBound19969.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound19969.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19985 .coefficient)
      LeftAuthority19982.bound (LeftAuthority19982.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority19982.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound19969.bound, LeftAuthority19982.bound]
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19969.bound, LeftAuthority19982.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound19969.actual selector witness, LeftAuthority19982.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound19986

namespace LeftBound19989
def owner : Owner := ⟨.program ⟨214⟩, ⟨15675⟩⟩
def transferEvent : Nat := 19989
def frameStart : Nat := 19930
def rule : BoundRule := .identity (.predecessor 0 19988 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19988 .coefficient)
      LeftBound19986.bound (LeftBound19986.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound19986.derived selector witness)

def rawBound : CoeffClass := LeftBound19986.bound
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19986.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound19986.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound19989

namespace LeftBound19995
def owner : Owner := ⟨.program ⟨214⟩, ⟨15676⟩⟩
def transferEvent : Nat := 19995
def frameStart : Nat := 19930
def rule : BoundRule := .product (.predecessor 0 19993 .coefficient) (.predecessor 1 19994 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19993 .coefficient)
      LeftAuthority19991.bound (LeftAuthority19991.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events078.exact19992RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19991.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19991.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19994 .coefficient)
      LeftBound19989.bound (LeftBound19989.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events078.exact19990RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19989.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19989.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority19991.bound LeftBound19989.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19991.bound, LeftBound19989.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority19991.actual selector witness) * (LeftBound19989.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound19995

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
