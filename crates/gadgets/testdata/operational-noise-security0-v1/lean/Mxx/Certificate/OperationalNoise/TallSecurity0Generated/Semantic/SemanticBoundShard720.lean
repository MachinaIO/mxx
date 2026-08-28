import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard019
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard667
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard668
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard719

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound105088
def owner : Owner := ⟨.program ⟨214⟩, ⟨28696⟩⟩
def transferEvent : Nat := 105088
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 105086 .coefficient) (.predecessor 1 105087 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105086 .coefficient)
      LeftBound105081.bound (LeftBound105081.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events410.exact105085RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105081.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105081.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105087 .coefficient)
      LeftBound5638.bound (LeftBound5638.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5639RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5638.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5638.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound105081.bound LeftBound5638.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105081.bound, LeftBound5638.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound105081.actual selector witness) * (LeftBound5638.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105088

namespace LeftBound105089
def owner : Owner := ⟨.program ⟨214⟩, ⟨28696⟩⟩
def transferEvent : Nat := 105089
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
end LeftBound105089

namespace LeftBound105090
def owner : Owner := ⟨.program ⟨214⟩, ⟨28696⟩⟩
def transferEvent : Nat := 105090
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 105085 .summary) (.transfer 105089) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 105085 .summary)
      LeftBound105084.bound (LeftBound105084.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28695⟩⟩) (rawTerms := some (Proof.Events410.exact105085RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound105084.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 105089)
      LeftBound105089.bound (LeftBound105089.actual selector witness) := by
  exact .transfer (LeftBound105089.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound105084.bound LeftBound105089.bound
def bound : CoeffClass := .finite ⟨4742652258740286904787271680, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105084.bound, LeftBound105089.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound105084.actual selector witness) * (LeftBound105089.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105090

namespace LeftBound105105
def owner : Owner := ⟨.program ⟨214⟩, ⟨28477⟩⟩
def transferEvent : Nat := 105105
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 105103 .coefficient) (.predecessor 1 105104 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105103 .coefficient)
      LeftBound97658.bound (LeftBound97658.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events381.exact97662RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97658.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97658.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105104 .coefficient)
      LeftAuthority105101.bound (LeftAuthority105101.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events410.exact105102RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105101.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105101.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound97658.bound LeftAuthority105101.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97658.bound, LeftAuthority105101.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound97658.actual selector witness) * (LeftAuthority105101.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105105

namespace LeftBound105106
def owner : Owner := ⟨.program ⟨214⟩, ⟨28477⟩⟩
def transferEvent : Nat := 105106
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28475⟩⟩]⟩ [⟨.result 105102 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 105102 .coefficient)
      LeftAuthority105101.bound (LeftAuthority105101.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28475⟩⟩) (rawTerms := some (Proof.Events410.exact105102RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105101.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105101.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority105101.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105101.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority105101.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound105106

namespace LeftBound105107
def owner : Owner := ⟨.program ⟨214⟩, ⟨28477⟩⟩
def transferEvent : Nat := 105107
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 97662 .summary) (.transfer 105106) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 97662 .summary)
      LeftBound97661.bound (LeftBound97661.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25131⟩⟩) (rawTerms := some (Proof.Events381.exact97662RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound97661.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 105106)
      LeftBound105106.bound (LeftBound105106.actual selector witness) := by
  exact .transfer (LeftBound105106.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound97661.bound LeftBound105106.bound
def bound : CoeffClass := .finite ⟨1292202946798406336512, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97661.bound, LeftBound105106.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound97661.actual selector witness) * (LeftBound105106.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105107

namespace LeftBound105118
def owner : Owner := ⟨.program ⟨214⟩, ⟨21751⟩⟩
def transferEvent : Nat := 105118
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 105116 .coefficient) (.value (.predecessor 1 105117 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105116 .coefficient)
      LeftAuthority105114.bound (LeftAuthority105114.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events410.exact105115RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105114.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105114.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105117 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority105114.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105114.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority105114.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound105118

namespace LeftBound105122
def owner : Owner := ⟨.program ⟨214⟩, ⟨21752⟩⟩
def transferEvent : Nat := 105122
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 105120 .coefficient) (.predecessor 1 105121 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105120 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105121 .coefficient)
      LeftBound105118.bound (LeftBound105118.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events410.exact105119RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105118.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105118.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound105118.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound105118.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound105118.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105122

namespace LeftBound105123
def owner : Owner := ⟨.program ⟨214⟩, ⟨21752⟩⟩
def transferEvent : Nat := 105123
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21749⟩⟩]⟩ [⟨.result 105115 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 105115 .coefficient)
      LeftAuthority105114.bound (LeftAuthority105114.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21749⟩⟩) (rawTerms := some (Proof.Events410.exact105115RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105114.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105114.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority105114.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105114.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority105114.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound105123

namespace LeftBound105124
def owner : Owner := ⟨.program ⟨214⟩, ⟨21752⟩⟩
def transferEvent : Nat := 105124
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 105123) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 105123)
      LeftBound105123.bound (LeftBound105123.actual selector witness) := by
  exact .transfer (LeftBound105123.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound105123.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound105123.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound105123.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105124

namespace LeftBound105195
def owner : Owner := ⟨.program ⟨214⟩, ⟨16253⟩⟩
def transferEvent : Nat := 105195
def frameStart : Nat := 105168
def rule : BoundRule := .identity (.predecessor 0 105194 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105194 .coefficient)
      LeftAuthority105192.bound (LeftAuthority105192.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events410.exact105193RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105192.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105192.derived selector witness)

def rawBound : CoeffClass := LeftAuthority105192.bound
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105192.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority105192.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound105195

namespace LeftBound105212
def owner : Owner := ⟨.program ⟨214⟩, ⟨16329⟩⟩
def transferEvent : Nat := 105212
def frameStart : Nat := 105168
def rule : BoundRule := .sum [.predecessor 0 105210 .coefficient, .predecessor 1 105211 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105210 .coefficient)
      LeftBound105195.bound (LeftBound105195.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound105195.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105211 .coefficient)
      LeftAuthority105208.bound (LeftAuthority105208.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority105208.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound105195.bound, LeftAuthority105208.bound]
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105195.bound, LeftAuthority105208.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound105195.actual selector witness, LeftAuthority105208.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound105212

namespace LeftBound105215
def owner : Owner := ⟨.program ⟨214⟩, ⟨16330⟩⟩
def transferEvent : Nat := 105215
def frameStart : Nat := 105168
def rule : BoundRule := .identity (.predecessor 0 105214 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105214 .coefficient)
      LeftBound105212.bound (LeftBound105212.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound105212.derived selector witness)

def rawBound : CoeffClass := LeftBound105212.bound
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105212.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound105212.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound105215

namespace LeftBound105221
def owner : Owner := ⟨.program ⟨214⟩, ⟨16331⟩⟩
def transferEvent : Nat := 105221
def frameStart : Nat := 105168
def rule : BoundRule := .product (.predecessor 0 105219 .coefficient) (.predecessor 1 105220 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105219 .coefficient)
      LeftAuthority105217.bound (LeftAuthority105217.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105218RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105217.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105217.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105220 .coefficient)
      LeftBound105215.bound (LeftBound105215.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105216RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105215.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105215.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority105217.bound LeftBound105215.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105217.bound, LeftBound105215.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority105217.actual selector witness) * (LeftBound105215.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105221

namespace LeftBound105229
def owner : Owner := ⟨.program ⟨214⟩, ⟨16332⟩⟩
def transferEvent : Nat := 105229
def frameStart : Nat := 105168
def rule : BoundRule := .sum [.predecessor 0 105227 .coefficient, .predecessor 1 105228 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105227 .coefficient)
      LeftAuthority105225.bound (LeftAuthority105225.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105226RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105225.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105225.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105228 .coefficient)
      LeftBound105221.bound (LeftBound105221.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105223RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105221.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105221.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority105225.bound, LeftBound105221.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105225.bound, LeftBound105221.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority105225.actual selector witness, LeftBound105221.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound105229

namespace LeftBound105233
def owner : Owner := ⟨.program ⟨214⟩, ⟨28476⟩⟩
def transferEvent : Nat := 105233
def frameStart : Nat := 105168
def rule : BoundRule := .product (.predecessor 0 105231 .coefficient) (.predecessor 1 105232 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105231 .coefficient)
      LeftBound105229.bound (LeftBound105229.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105230RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105229.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105229.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105232 .coefficient)
      LeftAuthority105206.bound (LeftAuthority105206.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events410.exact105207RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105206.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105206.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound105229.bound LeftAuthority105206.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105229.bound, LeftAuthority105206.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound105229.actual selector witness) * (LeftAuthority105206.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105233

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
