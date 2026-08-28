import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard194

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound29206
def owner : Owner := ⟨.program ⟨214⟩, ⟨10707⟩⟩
def transferEvent : Nat := 29206
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 29201 .summary, .result 29171 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 29201 .summary)
      LeftBound29196.bound (LeftBound29196.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9524⟩⟩) (rawTerms := some (Proof.Events114.exact29201RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound29196.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 29171 .summary)
      LeftBound29168.bound (LeftBound29168.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10706⟩⟩) (rawTerms := some (Proof.Events113.exact29171RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound29168.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound29196.bound, LeftBound29168.bound]
def bound : CoeffClass := .finite ⟨95422912, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29196.bound, LeftBound29168.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound29196.actual selector witness, LeftBound29168.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound29206

namespace LeftBound29210
def owner : Owner := ⟨.program ⟨214⟩, ⟨25004⟩⟩
def transferEvent : Nat := 29210
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 29208 .coefficient) (.predecessor 1 29209 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29208 .coefficient)
      LeftBound29204.bound (LeftBound29204.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events114.exact29207RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29204.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29204.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29209 .coefficient)
      LeftAuthority29142.bound (LeftAuthority29142.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events113.exact29143RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29142.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29142.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound29204.bound LeftAuthority29142.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29204.bound, LeftAuthority29142.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound29204.actual selector witness) * (LeftAuthority29142.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29210

namespace LeftBound29211
def owner : Owner := ⟨.program ⟨214⟩, ⟨25004⟩⟩
def transferEvent : Nat := 29211
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25003⟩⟩]⟩ [⟨.result 29143 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 29143 .coefficient)
      LeftAuthority29142.bound (LeftAuthority29142.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25003⟩⟩) (rawTerms := some (Proof.Events113.exact29143RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29142.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29142.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority29142.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority29142.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority29142.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound29211

namespace LeftBound29212
def owner : Owner := ⟨.program ⟨214⟩, ⟨25004⟩⟩
def transferEvent : Nat := 29212
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 29207 .summary) (.transfer 29211) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 29207 .summary)
      LeftBound29206.bound (LeftBound29206.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10707⟩⟩) (rawTerms := some (Proof.Events114.exact29207RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound29206.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 29211)
      LeftBound29211.bound (LeftBound29211.actual selector witness) := by
  exact .transfer (LeftBound29211.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound29206.bound LeftBound29211.bound
def bound : CoeffClass := .finite ⟨350203613806592, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29206.bound, LeftBound29211.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound29206.actual selector witness) * (LeftBound29211.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29212

namespace LeftBound29223
def owner : Owner := ⟨.program ⟨214⟩, ⟨19110⟩⟩
def transferEvent : Nat := 29223
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 29221 .coefficient) (.value (.predecessor 1 29222 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29221 .coefficient)
      LeftAuthority29219.bound (LeftAuthority29219.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events114.exact29220RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29219.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29219.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29222 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority29219.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority29219.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority29219.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound29223

namespace LeftBound29227
def owner : Owner := ⟨.program ⟨214⟩, ⟨19111⟩⟩
def transferEvent : Nat := 29227
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 29225 .coefficient) (.predecessor 1 29226 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29225 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29226 .coefficient)
      LeftBound29223.bound (LeftBound29223.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events114.exact29224RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29223.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29223.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound29223.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound29223.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound29223.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29227

namespace LeftBound29228
def owner : Owner := ⟨.program ⟨214⟩, ⟨19111⟩⟩
def transferEvent : Nat := 29228
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19108⟩⟩]⟩ [⟨.result 29220 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 29220 .coefficient)
      LeftAuthority29219.bound (LeftAuthority29219.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19108⟩⟩) (rawTerms := some (Proof.Events114.exact29220RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29219.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29219.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority29219.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority29219.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority29219.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound29228

namespace LeftBound29229
def owner : Owner := ⟨.program ⟨214⟩, ⟨19111⟩⟩
def transferEvent : Nat := 29229
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 29228) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 29228)
      LeftBound29228.bound (LeftBound29228.actual selector witness) := by
  exact .transfer (LeftBound29228.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound29228.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound29228.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound29228.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29229

namespace LeftBound29308
def owner : Owner := ⟨.program ⟨214⟩, ⟨10701⟩⟩
def transferEvent : Nat := 29308
def frameStart : Nat := 29279
def rule : BoundRule := .product (.predecessor 0 29306 .coefficient) (.predecessor 1 29307 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29306 .coefficient)
      LeftAuthority29304.bound (LeftAuthority29304.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events114.exact29305RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29304.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29304.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29307 .coefficient)
      LeftAuthority29301.bound (LeftAuthority29301.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events114.exact29302RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29301.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29301.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority29304.bound LeftAuthority29301.bound
def bound : CoeffClass := .finite ⟨9, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority29304.bound, LeftAuthority29301.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority29304.actual selector witness) * (LeftAuthority29301.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29308

namespace LeftBound29312
def owner : Owner := ⟨.program ⟨214⟩, ⟨10702⟩⟩
def transferEvent : Nat := 29312
def frameStart : Nat := 29279
def rule : BoundRule := .identity (.predecessor 0 29311 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29311 .coefficient)
      LeftBound29308.bound (LeftBound29308.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events114.exact29310RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29308.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29308.derived selector witness)

def rawBound : CoeffClass := LeftBound29308.bound
def bound : CoeffClass := .finite ⟨9, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29308.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound29308.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound29312

namespace LeftBound29329
def owner : Owner := ⟨.program ⟨214⟩, ⟨10784⟩⟩
def transferEvent : Nat := 29329
def frameStart : Nat := 29279
def rule : BoundRule := .sum [.predecessor 0 29327 .coefficient, .predecessor 1 29328 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29327 .coefficient)
      LeftBound29312.bound (LeftBound29312.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound29312.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29328 .coefficient)
      LeftAuthority29325.bound (LeftAuthority29325.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority29325.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound29312.bound, LeftAuthority29325.bound]
def bound : CoeffClass := .finite ⟨9, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29312.bound, LeftAuthority29325.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound29312.actual selector witness, LeftAuthority29325.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound29329

namespace LeftBound29332
def owner : Owner := ⟨.program ⟨214⟩, ⟨10785⟩⟩
def transferEvent : Nat := 29332
def frameStart : Nat := 29279
def rule : BoundRule := .identity (.predecessor 0 29331 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29331 .coefficient)
      LeftBound29329.bound (LeftBound29329.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound29329.derived selector witness)

def rawBound : CoeffClass := LeftBound29329.bound
def bound : CoeffClass := .finite ⟨9, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29329.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound29329.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound29332

namespace LeftBound29338
def owner : Owner := ⟨.program ⟨214⟩, ⟨10786⟩⟩
def transferEvent : Nat := 29338
def frameStart : Nat := 29279
def rule : BoundRule := .product (.predecessor 0 29336 .coefficient) (.predecessor 1 29337 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29336 .coefficient)
      LeftAuthority29334.bound (LeftAuthority29334.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events114.exact29335RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29334.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29334.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29337 .coefficient)
      LeftBound29332.bound (LeftBound29332.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events114.exact29333RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29332.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29332.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority29334.bound LeftBound29332.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority29334.bound, LeftBound29332.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority29334.actual selector witness) * (LeftBound29332.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29338

namespace LeftBound29354
def owner : Owner := ⟨.program ⟨214⟩, ⟨7835⟩⟩
def transferEvent : Nat := 29354
def frameStart : Nat := 29279
def rule : BoundRule := .scale (.predecessor 0 29352 .coefficient) (.value (.predecessor 1 29353 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29352 .coefficient)
      LeftAuthority29350.bound (LeftAuthority29350.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events114.exact29351RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29350.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29350.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29353 .coefficient)
      LeftAuthority29341.bound (LeftAuthority29341.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority29341.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority29350.bound LeftAuthority29341.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority29350.bound, LeftAuthority29341.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority29350.actual selector witness) * (LeftAuthority29341.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound29354

namespace LeftBound29357
def owner : Owner := ⟨.program ⟨214⟩, ⟨6782⟩⟩
def transferEvent : Nat := 29357
def frameStart : Nat := 29279
def rule : BoundRule := .identity (.predecessor 0 29356 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29356 .coefficient)
      LeftAuthority29344.bound (LeftAuthority29344.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events114.exact29345RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29344.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29344.derived selector witness)

def rawBound : CoeffClass := LeftAuthority29344.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority29344.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority29344.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound29357

namespace LeftBound29361
def owner : Owner := ⟨.program ⟨214⟩, ⟨7836⟩⟩
def transferEvent : Nat := 29361
def frameStart : Nat := 29279
def rule : BoundRule := .product (.predecessor 0 29359 .coefficient) (.predecessor 1 29360 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29359 .coefficient)
      LeftBound29357.bound (LeftBound29357.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events114.exact29358RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29357.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29357.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29360 .coefficient)
      LeftBound29354.bound (LeftBound29354.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events114.exact29355RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29354.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29354.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound29357.bound LeftBound29354.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29357.bound, LeftBound29354.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound29357.actual selector witness) * (LeftBound29354.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29361

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
