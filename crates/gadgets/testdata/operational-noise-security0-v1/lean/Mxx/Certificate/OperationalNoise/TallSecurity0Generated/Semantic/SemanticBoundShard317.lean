import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard019
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard265
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard316

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound47799
def owner : Owner := ⟨.program ⟨214⟩, ⟨28759⟩⟩
def transferEvent : Nat := 47799
def frameStart : Nat := 47699
def rule : BoundRule := .sum [.predecessor 0 47797 .coefficient, .predecessor 1 47798 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47797 .coefficient)
      LeftBound47795.bound (LeftBound47795.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events186.exact47796RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47795.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47795.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47798 .coefficient)
      LeftBound47776.bound (LeftBound47776.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events186.exact47781RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47776.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47776.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound47795.bound, LeftBound47776.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound47795.bound, LeftBound47776.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound47795.actual selector witness, LeftBound47776.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound47799

namespace LeftBound47812
def owner : Owner := ⟨.program ⟨214⟩, ⟨28756⟩⟩
def transferEvent : Nat := 47812
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 47810 .coefficient, .predecessor 1 47811 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47810 .coefficient)
      LeftBound47641.bound (LeftBound47641.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events186.exact47809RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47641.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47641.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47811 .coefficient)
      LeftBound47624.bound (LeftBound47624.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events186.exact47631RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47624.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47624.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound47641.bound, LeftBound47624.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound47641.bound, LeftBound47624.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound47641.actual selector witness, LeftBound47624.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound47812

namespace LeftBound47815
def owner : Owner := ⟨.program ⟨214⟩, ⟨28756⟩⟩
def transferEvent : Nat := 47815
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 47809 .summary, .result 47631 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 47809 .summary)
      LeftBound47643.bound (LeftBound47643.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21915⟩⟩) (rawTerms := some (Proof.Events186.exact47809RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound47643.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 47631 .summary)
      LeftBound47626.bound (LeftBound47626.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28755⟩⟩) (rawTerms := some (Proof.Events186.exact47631RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound47626.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound47643.bound, LeftBound47626.bound]
def bound : CoeffClass := .finite ⟨1292270185944771604480, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound47643.bound, LeftBound47626.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound47643.actual selector witness, LeftBound47626.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound47815

namespace LeftBound47819
def owner : Owner := ⟨.program ⟨214⟩, ⟨28757⟩⟩
def transferEvent : Nat := 47819
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 47817 .coefficient) (.predecessor 1 47818 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47817 .coefficient)
      LeftBound47812.bound (LeftBound47812.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events186.exact47816RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47812.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47812.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47818 .coefficient)
      LeftBound5638.bound (LeftBound5638.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5639RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5638.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5638.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound47812.bound LeftBound5638.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound47812.bound, LeftBound5638.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound47812.actual selector witness) * (LeftBound5638.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound47819

namespace LeftBound47820
def owner : Owner := ⟨.program ⟨214⟩, ⟨28757⟩⟩
def transferEvent : Nat := 47820
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6673⟩⟩]⟩ [⟨.result 5635 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5635 .coefficient)
      LeftAuthority5634.bound (LeftAuthority5634.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6673⟩⟩) (rawTerms := some (Proof.Events022.exact5635RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5634.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5634.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5634.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5634.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5634.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound47820

namespace LeftBound47821
def owner : Owner := ⟨.program ⟨214⟩, ⟨28757⟩⟩
def transferEvent : Nat := 47821
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 47816 .summary) (.transfer 47820) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 47816 .summary)
      LeftBound47815.bound (LeftBound47815.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28756⟩⟩) (rawTerms := some (Proof.Events186.exact47816RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound47815.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 47820)
      LeftBound47820.bound (LeftBound47820.actual selector witness) := by
  exact .transfer (LeftBound47820.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound47815.bound LeftBound47820.bound
def bound : CoeffClass := .finite ⟨4742652258740286904787271680, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound47815.bound, LeftBound47820.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound47815.actual selector witness) * (LeftBound47820.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound47821

namespace LeftBound47836
def owner : Owner := ⟨.program ⟨214⟩, ⟨28538⟩⟩
def transferEvent : Nat := 47836
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 47834 .coefficient) (.predecessor 1 47835 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47834 .coefficient)
      LeftBound39693.bound (LeftBound39693.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events155.exact39697RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39693.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39693.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47835 .coefficient)
      LeftAuthority47832.bound (LeftAuthority47832.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events186.exact47833RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47832.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47832.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound39693.bound LeftAuthority47832.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39693.bound, LeftAuthority47832.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound39693.actual selector witness) * (LeftAuthority47832.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound47836

namespace LeftBound47837
def owner : Owner := ⟨.program ⟨214⟩, ⟨28538⟩⟩
def transferEvent : Nat := 47837
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28536⟩⟩]⟩ [⟨.result 47833 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 47833 .coefficient)
      LeftAuthority47832.bound (LeftAuthority47832.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28536⟩⟩) (rawTerms := some (Proof.Events186.exact47833RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47832.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47832.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority47832.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority47832.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority47832.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound47837

namespace LeftBound47838
def owner : Owner := ⟨.program ⟨214⟩, ⟨28538⟩⟩
def transferEvent : Nat := 47838
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 39697 .summary) (.transfer 47837) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 39697 .summary)
      LeftBound39696.bound (LeftBound39696.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25154⟩⟩) (rawTerms := some (Proof.Events155.exact39697RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound39696.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 47837)
      LeftBound47837.bound (LeftBound47837.actual selector witness) := by
  exact .transfer (LeftBound47837.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound39696.bound LeftBound47837.bound
def bound : CoeffClass := .finite ⟨1292202946798406336512, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39696.bound, LeftBound47837.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound39696.actual selector witness) * (LeftBound47837.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound47838

namespace LeftBound47849
def owner : Owner := ⟨.program ⟨214⟩, ⟨21770⟩⟩
def transferEvent : Nat := 47849
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 47847 .coefficient) (.value (.predecessor 1 47848 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47847 .coefficient)
      LeftAuthority47845.bound (LeftAuthority47845.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events186.exact47846RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47845.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47845.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47848 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority47845.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority47845.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority47845.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound47849

namespace LeftBound47853
def owner : Owner := ⟨.program ⟨214⟩, ⟨21771⟩⟩
def transferEvent : Nat := 47853
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 47851 .coefficient) (.predecessor 1 47852 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47851 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47852 .coefficient)
      LeftBound47849.bound (LeftBound47849.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events186.exact47850RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47849.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47849.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound47849.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound47849.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound47849.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound47853

namespace LeftBound47854
def owner : Owner := ⟨.program ⟨214⟩, ⟨21771⟩⟩
def transferEvent : Nat := 47854
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21768⟩⟩]⟩ [⟨.result 47846 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 47846 .coefficient)
      LeftAuthority47845.bound (LeftAuthority47845.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21768⟩⟩) (rawTerms := some (Proof.Events186.exact47846RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47845.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47845.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority47845.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority47845.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority47845.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound47854

namespace LeftBound47855
def owner : Owner := ⟨.program ⟨214⟩, ⟨21771⟩⟩
def transferEvent : Nat := 47855
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 47854) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 47854)
      LeftBound47854.bound (LeftBound47854.actual selector witness) := by
  exact .transfer (LeftBound47854.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound47854.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound47854.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound47854.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound47855

namespace LeftBound47950
def owner : Owner := ⟨.program ⟨214⟩, ⟨16271⟩⟩
def transferEvent : Nat := 47950
def frameStart : Nat := 47911
def rule : BoundRule := .identity (.predecessor 0 47949 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47949 .coefficient)
      LeftAuthority47947.bound (LeftAuthority47947.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events187.exact47948RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47947.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47947.derived selector witness)

def rawBound : CoeffClass := LeftAuthority47947.bound
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority47947.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority47947.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound47950

namespace LeftBound47967
def owner : Owner := ⟨.program ⟨214⟩, ⟨16345⟩⟩
def transferEvent : Nat := 47967
def frameStart : Nat := 47911
def rule : BoundRule := .sum [.predecessor 0 47965 .coefficient, .predecessor 1 47966 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47965 .coefficient)
      LeftBound47950.bound (LeftBound47950.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound47950.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47966 .coefficient)
      LeftAuthority47963.bound (LeftAuthority47963.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority47963.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound47950.bound, LeftAuthority47963.bound]
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound47950.bound, LeftAuthority47963.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound47950.actual selector witness, LeftAuthority47963.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound47967

namespace LeftBound47970
def owner : Owner := ⟨.program ⟨214⟩, ⟨16346⟩⟩
def transferEvent : Nat := 47970
def frameStart : Nat := 47911
def rule : BoundRule := .identity (.predecessor 0 47969 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47969 .coefficient)
      LeftBound47967.bound (LeftBound47967.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound47967.derived selector witness)

def rawBound : CoeffClass := LeftBound47967.bound
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound47967.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound47967.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound47970

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
