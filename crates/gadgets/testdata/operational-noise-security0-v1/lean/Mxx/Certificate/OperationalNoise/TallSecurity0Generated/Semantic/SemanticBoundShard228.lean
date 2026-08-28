import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard021
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard227

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound35136
def owner : Owner := ⟨.program ⟨214⟩, ⟨20479⟩⟩
def transferEvent : Nat := 35136
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 35134 .coefficient) (.predecessor 1 35135 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35134 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35135 .coefficient)
      LeftBound35132.bound (LeftBound35132.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events137.exact35133RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35132.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35132.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound35132.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound35132.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound35132.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound35136

namespace LeftBound35137
def owner : Owner := ⟨.program ⟨214⟩, ⟨20479⟩⟩
def transferEvent : Nat := 35137
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20476⟩⟩]⟩ [⟨.result 35129 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 35129 .coefficient)
      LeftAuthority35128.bound (LeftAuthority35128.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20476⟩⟩) (rawTerms := some (Proof.Events137.exact35129RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35128.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35128.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority35128.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority35128.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority35128.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound35137

namespace LeftBound35138
def owner : Owner := ⟨.program ⟨214⟩, ⟨20479⟩⟩
def transferEvent : Nat := 35138
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 35137) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 35137)
      LeftBound35137.bound (LeftBound35137.actual selector witness) := by
  exact .transfer (LeftBound35137.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound35137.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound35137.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound35137.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound35138

namespace LeftBound35233
def owner : Owner := ⟨.program ⟨214⟩, ⟨14966⟩⟩
def transferEvent : Nat := 35233
def frameStart : Nat := 35194
def rule : BoundRule := .identity (.predecessor 0 35232 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35232 .coefficient)
      LeftAuthority35230.bound (LeftAuthority35230.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events137.exact35231RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35230.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35230.derived selector witness)

def rawBound : CoeffClass := LeftAuthority35230.bound
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority35230.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority35230.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound35233

namespace LeftBound35250
def owner : Owner := ⟨.program ⟨214⟩, ⟨15005⟩⟩
def transferEvent : Nat := 35250
def frameStart : Nat := 35194
def rule : BoundRule := .sum [.predecessor 0 35248 .coefficient, .predecessor 1 35249 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35248 .coefficient)
      LeftBound35233.bound (LeftBound35233.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound35233.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35249 .coefficient)
      LeftAuthority35246.bound (LeftAuthority35246.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority35246.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35233.bound, LeftAuthority35246.bound]
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35233.bound, LeftAuthority35246.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35233.actual selector witness, LeftAuthority35246.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35250

namespace LeftBound35253
def owner : Owner := ⟨.program ⟨214⟩, ⟨15006⟩⟩
def transferEvent : Nat := 35253
def frameStart : Nat := 35194
def rule : BoundRule := .identity (.predecessor 0 35252 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35252 .coefficient)
      LeftBound35250.bound (LeftBound35250.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound35250.derived selector witness)

def rawBound : CoeffClass := LeftBound35250.bound
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35250.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound35250.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound35253

namespace LeftBound35259
def owner : Owner := ⟨.program ⟨214⟩, ⟨15007⟩⟩
def transferEvent : Nat := 35259
def frameStart : Nat := 35194
def rule : BoundRule := .product (.predecessor 0 35257 .coefficient) (.predecessor 1 35258 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35257 .coefficient)
      LeftAuthority35255.bound (LeftAuthority35255.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events137.exact35256RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35255.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35255.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35258 .coefficient)
      LeftBound35253.bound (LeftBound35253.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events137.exact35254RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35253.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35253.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority35255.bound LeftBound35253.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority35255.bound, LeftBound35253.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority35255.actual selector witness) * (LeftBound35253.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound35259

namespace LeftBound35267
def owner : Owner := ⟨.program ⟨214⟩, ⟨15008⟩⟩
def transferEvent : Nat := 35267
def frameStart : Nat := 35194
def rule : BoundRule := .sum [.predecessor 0 35265 .coefficient, .predecessor 1 35266 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35265 .coefficient)
      LeftAuthority35263.bound (LeftAuthority35263.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events137.exact35264RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35263.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35263.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35266 .coefficient)
      LeftBound35259.bound (LeftBound35259.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events137.exact35261RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35259.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35259.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority35263.bound, LeftBound35259.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority35263.bound, LeftBound35259.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority35263.actual selector witness, LeftBound35259.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35267

namespace LeftBound35271
def owner : Owner := ⟨.program ⟨214⟩, ⟨26597⟩⟩
def transferEvent : Nat := 35271
def frameStart : Nat := 35194
def rule : BoundRule := .product (.predecessor 0 35269 .coefficient) (.predecessor 1 35270 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35269 .coefficient)
      LeftBound35267.bound (LeftBound35267.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events137.exact35268RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35267.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35267.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35270 .coefficient)
      LeftAuthority35244.bound (LeftAuthority35244.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events137.exact35245RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35244.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35244.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound35267.bound LeftAuthority35244.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35267.bound, LeftAuthority35244.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound35267.actual selector witness) * (LeftAuthority35244.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound35271

namespace LeftBound35282
def owner : Owner := ⟨.program ⟨214⟩, ⟨15065⟩⟩
def transferEvent : Nat := 35282
def frameStart : Nat := 35194
def rule : BoundRule := .product (.predecessor 0 35280 .coefficient) (.predecessor 1 35281 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35280 .coefficient)
      LeftAuthority35255.bound (LeftAuthority35255.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events137.exact35256RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35255.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35255.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35281 .coefficient)
      LeftAuthority35278.bound (LeftAuthority35278.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events137.exact35279RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35278.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35278.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority35255.bound LeftAuthority35278.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority35255.bound, LeftAuthority35278.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority35255.actual selector witness) * (LeftAuthority35278.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound35282

namespace LeftBound35290
def owner : Owner := ⟨.program ⟨214⟩, ⟨15066⟩⟩
def transferEvent : Nat := 35290
def frameStart : Nat := 35194
def rule : BoundRule := .sum [.predecessor 0 35288 .coefficient, .predecessor 1 35289 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35288 .coefficient)
      LeftAuthority35286.bound (LeftAuthority35286.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events137.exact35287RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35286.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35286.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35289 .coefficient)
      LeftBound35282.bound (LeftBound35282.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events137.exact35284RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35282.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35282.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority35286.bound, LeftBound35282.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority35286.bound, LeftBound35282.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority35286.actual selector witness, LeftBound35282.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35290

namespace LeftBound35294
def owner : Owner := ⟨.program ⟨214⟩, ⟨26602⟩⟩
def transferEvent : Nat := 35294
def frameStart : Nat := 35194
def rule : BoundRule := .sum [.predecessor 0 35292 .coefficient, .predecessor 1 35293 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35292 .coefficient)
      LeftBound35290.bound (LeftBound35290.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events137.exact35291RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35290.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35290.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35293 .coefficient)
      LeftBound35271.bound (LeftBound35271.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events137.exact35276RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35271.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35271.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35290.bound, LeftBound35271.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35290.bound, LeftBound35271.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35290.actual selector witness, LeftBound35271.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35294

namespace LeftBound35307
def owner : Owner := ⟨.program ⟨214⟩, ⟨26599⟩⟩
def transferEvent : Nat := 35307
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 35305 .coefficient, .predecessor 1 35306 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35305 .coefficient)
      LeftBound35136.bound (LeftBound35136.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events137.exact35304RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35136.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35136.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35306 .coefficient)
      LeftBound35119.bound (LeftBound35119.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events137.exact35126RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35119.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35119.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35136.bound, LeftBound35119.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35136.bound, LeftBound35119.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35136.actual selector witness, LeftBound35119.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35307

namespace LeftBound35310
def owner : Owner := ⟨.program ⟨214⟩, ⟨26599⟩⟩
def transferEvent : Nat := 35310
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 35304 .summary, .result 35126 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 35304 .summary)
      LeftBound35138.bound (LeftBound35138.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20479⟩⟩) (rawTerms := some (Proof.Events137.exact35304RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound35138.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 35126 .summary)
      LeftBound35121.bound (LeftBound35121.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26598⟩⟩) (rawTerms := some (Proof.Events137.exact35126RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound35121.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35138.bound, LeftBound35121.bound]
def bound : CoeffClass := .finite ⟨1291900380601931935744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35138.bound, LeftBound35121.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35138.actual selector witness, LeftBound35121.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35310

namespace LeftBound35314
def owner : Owner := ⟨.program ⟨214⟩, ⟨26600⟩⟩
def transferEvent : Nat := 35314
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 35312 .coefficient) (.predecessor 1 35313 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35312 .coefficient)
      LeftBound35307.bound (LeftBound35307.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events137.exact35311RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35307.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35307.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35313 .coefficient)
      LeftBound5838.bound (LeftBound5838.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5839RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5838.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5838.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound35307.bound LeftBound5838.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35307.bound, LeftBound5838.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound35307.actual selector witness) * (LeftBound5838.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound35314

namespace LeftBound35315
def owner : Owner := ⟨.program ⟨214⟩, ⟨26600⟩⟩
def transferEvent : Nat := 35315
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩ [⟨.result 5835 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5835 .coefficient)
      LeftAuthority5834.bound (LeftAuthority5834.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6671⟩⟩) (rawTerms := some (Proof.Events022.exact5835RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5834.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5834.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5834.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5834.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5834.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound35315

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
