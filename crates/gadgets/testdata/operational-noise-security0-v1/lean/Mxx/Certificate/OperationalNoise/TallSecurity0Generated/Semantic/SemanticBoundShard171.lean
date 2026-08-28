import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard169
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard170

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound26035
def owner : Owner := ⟨.program ⟨214⟩, ⟨26160⟩⟩
def transferEvent : Nat := 26035
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 26029 .summary, .result 25843 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 26029 .summary)
      LeftBound25855.bound (LeftBound25855.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19615⟩⟩) (rawTerms := some (Proof.Events101.exact26029RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound25855.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 25843 .summary)
      LeftBound25838.bound (LeftBound25838.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26159⟩⟩) (rawTerms := some (Proof.Events100.exact25843RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound25838.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound25855.bound, LeftBound25838.bound]
def bound : CoeffClass := .finite ⟨352072932929536, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25855.bound, LeftBound25838.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound25855.actual selector witness, LeftBound25838.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26035

namespace LeftBound26039
def owner : Owner := ⟨.program ⟨214⟩, ⟨28124⟩⟩
def transferEvent : Nat := 26039
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 26037 .coefficient) (.predecessor 1 26038 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26037 .coefficient)
      LeftBound26032.bound (LeftBound26032.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events101.exact26036RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26032.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26032.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26038 .coefficient)
      LeftAuthority25758.bound (LeftAuthority25758.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events100.exact25759RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25758.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25758.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound26032.bound LeftAuthority25758.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26032.bound, LeftAuthority25758.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound26032.actual selector witness) * (LeftAuthority25758.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound26039

namespace LeftBound26040
def owner : Owner := ⟨.program ⟨214⟩, ⟨28124⟩⟩
def transferEvent : Nat := 26040
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28122⟩⟩]⟩ [⟨.result 25759 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 25759 .coefficient)
      LeftAuthority25758.bound (LeftAuthority25758.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28122⟩⟩) (rawTerms := some (Proof.Events100.exact25759RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25758.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25758.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority25758.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority25758.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority25758.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound26040

namespace LeftBound26041
def owner : Owner := ⟨.program ⟨214⟩, ⟨28124⟩⟩
def transferEvent : Nat := 26041
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 26036 .summary) (.transfer 26040) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 26036 .summary)
      LeftBound26035.bound (LeftBound26035.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26160⟩⟩) (rawTerms := some (Proof.Events101.exact26036RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound26035.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 26040)
      LeftBound26040.bound (LeftBound26040.actual selector witness) := by
  exact .transfer (LeftBound26040.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound26035.bound LeftBound26040.bound
def bound : CoeffClass := .finite ⟨1292113297018323992576, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26035.bound, LeftBound26040.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound26035.actual selector witness) * (LeftBound26040.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound26041

namespace LeftBound26052
def owner : Owner := ⟨.program ⟨214⟩, ⟨21558⟩⟩
def transferEvent : Nat := 26052
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 26050 .coefficient) (.value (.predecessor 1 26051 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26050 .coefficient)
      LeftAuthority26048.bound (LeftAuthority26048.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events101.exact26049RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26048.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26048.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26051 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority26048.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority26048.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority26048.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound26052

namespace LeftBound26056
def owner : Owner := ⟨.program ⟨214⟩, ⟨21559⟩⟩
def transferEvent : Nat := 26056
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 26054 .coefficient) (.predecessor 1 26055 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26054 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26055 .coefficient)
      LeftBound26052.bound (LeftBound26052.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events101.exact26053RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26052.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26052.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound26052.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound26052.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound26052.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound26056

namespace LeftBound26057
def owner : Owner := ⟨.program ⟨214⟩, ⟨21559⟩⟩
def transferEvent : Nat := 26057
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21556⟩⟩]⟩ [⟨.result 26049 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 26049 .coefficient)
      LeftAuthority26048.bound (LeftAuthority26048.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21556⟩⟩) (rawTerms := some (Proof.Events101.exact26049RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26048.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26048.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority26048.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority26048.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority26048.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound26057

namespace LeftBound26058
def owner : Owner := ⟨.program ⟨214⟩, ⟨21559⟩⟩
def transferEvent : Nat := 26058
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 26057) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 26057)
      LeftBound26057.bound (LeftBound26057.actual selector witness) := by
  exact .transfer (LeftBound26057.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound26057.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound26057.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound26057.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound26058

namespace LeftBound26153
def owner : Owner := ⟨.program ⟨214⟩, ⟨16072⟩⟩
def transferEvent : Nat := 26153
def frameStart : Nat := 26114
def rule : BoundRule := .identity (.predecessor 0 26152 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26152 .coefficient)
      LeftAuthority26150.bound (LeftAuthority26150.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events102.exact26151RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26150.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26150.derived selector witness)

def rawBound : CoeffClass := LeftAuthority26150.bound
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority26150.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority26150.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound26153

namespace LeftBound26170
def owner : Owner := ⟨.program ⟨214⟩, ⟨16146⟩⟩
def transferEvent : Nat := 26170
def frameStart : Nat := 26114
def rule : BoundRule := .sum [.predecessor 0 26168 .coefficient, .predecessor 1 26169 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26168 .coefficient)
      LeftBound26153.bound (LeftBound26153.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound26153.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26169 .coefficient)
      LeftAuthority26166.bound (LeftAuthority26166.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority26166.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26153.bound, LeftAuthority26166.bound]
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26153.bound, LeftAuthority26166.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound26153.actual selector witness, LeftAuthority26166.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26170

namespace LeftBound26173
def owner : Owner := ⟨.program ⟨214⟩, ⟨16147⟩⟩
def transferEvent : Nat := 26173
def frameStart : Nat := 26114
def rule : BoundRule := .identity (.predecessor 0 26172 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26172 .coefficient)
      LeftBound26170.bound (LeftBound26170.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound26170.derived selector witness)

def rawBound : CoeffClass := LeftBound26170.bound
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26170.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound26170.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound26173

namespace LeftBound26179
def owner : Owner := ⟨.program ⟨214⟩, ⟨16148⟩⟩
def transferEvent : Nat := 26179
def frameStart : Nat := 26114
def rule : BoundRule := .product (.predecessor 0 26177 .coefficient) (.predecessor 1 26178 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26177 .coefficient)
      LeftAuthority26175.bound (LeftAuthority26175.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events102.exact26176RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26175.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26175.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26178 .coefficient)
      LeftBound26173.bound (LeftBound26173.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events102.exact26174RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26173.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26173.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority26175.bound LeftBound26173.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority26175.bound, LeftBound26173.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority26175.actual selector witness) * (LeftBound26173.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound26179

namespace LeftBound26187
def owner : Owner := ⟨.program ⟨214⟩, ⟨16149⟩⟩
def transferEvent : Nat := 26187
def frameStart : Nat := 26114
def rule : BoundRule := .sum [.predecessor 0 26185 .coefficient, .predecessor 1 26186 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26185 .coefficient)
      LeftAuthority26183.bound (LeftAuthority26183.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events102.exact26184RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26183.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26183.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26186 .coefficient)
      LeftBound26179.bound (LeftBound26179.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events102.exact26181RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26179.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26179.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority26183.bound, LeftBound26179.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority26183.bound, LeftBound26179.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority26183.actual selector witness, LeftBound26179.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26187

namespace LeftBound26191
def owner : Owner := ⟨.program ⟨214⟩, ⟨28123⟩⟩
def transferEvent : Nat := 26191
def frameStart : Nat := 26114
def rule : BoundRule := .product (.predecessor 0 26189 .coefficient) (.predecessor 1 26190 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26189 .coefficient)
      LeftBound26187.bound (LeftBound26187.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events102.exact26188RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26187.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26187.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26190 .coefficient)
      LeftAuthority26164.bound (LeftAuthority26164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events102.exact26165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26164.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26164.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound26187.bound LeftAuthority26164.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26187.bound, LeftAuthority26164.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound26187.actual selector witness) * (LeftAuthority26164.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound26191

namespace LeftBound26202
def owner : Owner := ⟨.program ⟨214⟩, ⟨16115⟩⟩
def transferEvent : Nat := 26202
def frameStart : Nat := 26114
def rule : BoundRule := .product (.predecessor 0 26200 .coefficient) (.predecessor 1 26201 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26200 .coefficient)
      LeftAuthority26175.bound (LeftAuthority26175.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events102.exact26176RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26175.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26175.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26201 .coefficient)
      LeftAuthority26198.bound (LeftAuthority26198.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events102.exact26199RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26198.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26198.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority26175.bound LeftAuthority26198.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority26175.bound, LeftAuthority26198.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority26175.actual selector witness) * (LeftAuthority26198.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound26202

namespace LeftBound26210
def owner : Owner := ⟨.program ⟨214⟩, ⟨16116⟩⟩
def transferEvent : Nat := 26210
def frameStart : Nat := 26114
def rule : BoundRule := .sum [.predecessor 0 26208 .coefficient, .predecessor 1 26209 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26208 .coefficient)
      LeftAuthority26206.bound (LeftAuthority26206.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events102.exact26207RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26206.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26206.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26209 .coefficient)
      LeftBound26202.bound (LeftBound26202.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events102.exact26204RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26202.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26202.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority26206.bound, LeftBound26202.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority26206.bound, LeftBound26202.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority26206.actual selector witness, LeftBound26202.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26210

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
