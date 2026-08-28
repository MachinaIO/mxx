import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard689
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard727

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound106218
def owner : Owner := ⟨.program ⟨214⟩, ⟨27394⟩⟩
def transferEvent : Nat := 106218
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 106213 .summary) (.transfer 106217) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 106213 .summary)
      LeftBound106212.bound (LeftBound106212.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27393⟩⟩) (rawTerms := some (Proof.Events414.exact106213RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound106212.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 106217)
      LeftBound106217.bound (LeftBound106217.actual selector witness) := by
  exact .transfer (LeftBound106217.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound106212.bound LeftBound106217.bound
def bound : CoeffClass := .finite ⟨4741665210358390854099402752, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106212.bound, LeftBound106217.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound106212.actual selector witness) * (LeftBound106217.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106218

namespace LeftBound106233
def owner : Owner := ⟨.program ⟨214⟩, ⟨27175⟩⟩
def transferEvent : Nat := 106233
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 106231 .coefficient) (.predecessor 1 106232 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106231 .coefficient)
      LeftBound100262.bound (LeftBound100262.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events391.exact100266RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100262.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100262.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106232 .coefficient)
      LeftAuthority106229.bound (LeftAuthority106229.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events414.exact106230RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106229.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106229.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound100262.bound LeftAuthority106229.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100262.bound, LeftAuthority106229.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound100262.actual selector witness) * (LeftAuthority106229.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106233

namespace LeftBound106234
def owner : Owner := ⟨.program ⟨214⟩, ⟨27175⟩⟩
def transferEvent : Nat := 106234
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27173⟩⟩]⟩ [⟨.result 106230 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 106230 .coefficient)
      LeftAuthority106229.bound (LeftAuthority106229.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27173⟩⟩) (rawTerms := some (Proof.Events414.exact106230RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106229.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106229.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority106229.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106229.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority106229.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound106234

namespace LeftBound106235
def owner : Owner := ⟨.program ⟨214⟩, ⟨27175⟩⟩
def transferEvent : Nat := 106235
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 100266 .summary) (.transfer 106234) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 100266 .summary)
      LeftBound100265.bound (LeftBound100265.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25824⟩⟩) (rawTerms := some (Proof.Events391.exact100266RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound100265.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 106234)
      LeftBound106234.bound (LeftBound106234.actual selector witness) := by
  exact .transfer (LeftBound106234.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound100265.bound LeftBound106234.bound
def bound : CoeffClass := .finite ⟨1291978822348200476672, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100265.bound, LeftBound106234.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound100265.actual selector witness) * (LeftBound106234.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106235

namespace LeftBound106246
def owner : Owner := ⟨.program ⟨214⟩, ⟨20887⟩⟩
def transferEvent : Nat := 106246
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 106244 .coefficient) (.value (.predecessor 1 106245 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106244 .coefficient)
      LeftAuthority106242.bound (LeftAuthority106242.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106243RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106242.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106242.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106245 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority106242.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106242.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority106242.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound106246

namespace LeftBound106250
def owner : Owner := ⟨.program ⟨214⟩, ⟨20888⟩⟩
def transferEvent : Nat := 106250
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 106248 .coefficient) (.predecessor 1 106249 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106248 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106249 .coefficient)
      LeftBound106246.bound (LeftBound106246.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106247RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106246.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106246.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound106246.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound106246.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound106246.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106250

namespace LeftBound106251
def owner : Owner := ⟨.program ⟨214⟩, ⟨20888⟩⟩
def transferEvent : Nat := 106251
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20885⟩⟩]⟩ [⟨.result 106243 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 106243 .coefficient)
      LeftAuthority106242.bound (LeftAuthority106242.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20885⟩⟩) (rawTerms := some (Proof.Events415.exact106243RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106242.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106242.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority106242.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106242.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority106242.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound106251

namespace LeftBound106252
def owner : Owner := ⟨.program ⟨214⟩, ⟨20888⟩⟩
def transferEvent : Nat := 106252
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 106251) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 106251)
      LeftBound106251.bound (LeftBound106251.actual selector witness) := by
  exact .transfer (LeftBound106251.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound106251.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound106251.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound106251.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106252

namespace LeftBound106323
def owner : Owner := ⟨.program ⟨214⟩, ⟨15574⟩⟩
def transferEvent : Nat := 106323
def frameStart : Nat := 106296
def rule : BoundRule := .identity (.predecessor 0 106322 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106322 .coefficient)
      LeftAuthority106320.bound (LeftAuthority106320.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106321RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106320.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106320.derived selector witness)

def rawBound : CoeffClass := LeftAuthority106320.bound
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106320.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority106320.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound106323

namespace LeftBound106340
def owner : Owner := ⟨.program ⟨214⟩, ⟨15650⟩⟩
def transferEvent : Nat := 106340
def frameStart : Nat := 106296
def rule : BoundRule := .sum [.predecessor 0 106338 .coefficient, .predecessor 1 106339 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106338 .coefficient)
      LeftBound106323.bound (LeftBound106323.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound106323.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106339 .coefficient)
      LeftAuthority106336.bound (LeftAuthority106336.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority106336.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound106323.bound, LeftAuthority106336.bound]
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106323.bound, LeftAuthority106336.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound106323.actual selector witness, LeftAuthority106336.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound106340

namespace LeftBound106343
def owner : Owner := ⟨.program ⟨214⟩, ⟨15651⟩⟩
def transferEvent : Nat := 106343
def frameStart : Nat := 106296
def rule : BoundRule := .identity (.predecessor 0 106342 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106342 .coefficient)
      LeftBound106340.bound (LeftBound106340.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound106340.derived selector witness)

def rawBound : CoeffClass := LeftBound106340.bound
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106340.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound106340.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound106343

namespace LeftBound106349
def owner : Owner := ⟨.program ⟨214⟩, ⟨15652⟩⟩
def transferEvent : Nat := 106349
def frameStart : Nat := 106296
def rule : BoundRule := .product (.predecessor 0 106347 .coefficient) (.predecessor 1 106348 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106347 .coefficient)
      LeftAuthority106345.bound (LeftAuthority106345.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106346RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106345.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106345.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106348 .coefficient)
      LeftBound106343.bound (LeftBound106343.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106344RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106343.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106343.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority106345.bound LeftBound106343.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106345.bound, LeftBound106343.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority106345.actual selector witness) * (LeftBound106343.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106349

namespace LeftBound106357
def owner : Owner := ⟨.program ⟨214⟩, ⟨15653⟩⟩
def transferEvent : Nat := 106357
def frameStart : Nat := 106296
def rule : BoundRule := .sum [.predecessor 0 106355 .coefficient, .predecessor 1 106356 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106355 .coefficient)
      LeftAuthority106353.bound (LeftAuthority106353.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106354RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106353.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106353.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106356 .coefficient)
      LeftBound106349.bound (LeftBound106349.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106351RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106349.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106349.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority106353.bound, LeftBound106349.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106353.bound, LeftBound106349.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority106353.actual selector witness, LeftBound106349.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound106357

namespace LeftBound106361
def owner : Owner := ⟨.program ⟨214⟩, ⟨27174⟩⟩
def transferEvent : Nat := 106361
def frameStart : Nat := 106296
def rule : BoundRule := .product (.predecessor 0 106359 .coefficient) (.predecessor 1 106360 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106359 .coefficient)
      LeftBound106357.bound (LeftBound106357.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106358RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106357.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106357.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106360 .coefficient)
      LeftAuthority106334.bound (LeftAuthority106334.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106335RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106334.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106334.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound106357.bound LeftAuthority106334.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106357.bound, LeftAuthority106334.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound106357.actual selector witness) * (LeftAuthority106334.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106361

namespace LeftBound106372
def owner : Owner := ⟨.program ⟨214⟩, ⟨17798⟩⟩
def transferEvent : Nat := 106372
def frameStart : Nat := 106296
def rule : BoundRule := .product (.predecessor 0 106370 .coefficient) (.predecessor 1 106371 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106370 .coefficient)
      LeftAuthority106345.bound (LeftAuthority106345.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106346RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106345.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106345.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106371 .coefficient)
      LeftAuthority106368.bound (LeftAuthority106368.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106369RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106368.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106368.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority106345.bound LeftAuthority106368.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106345.bound, LeftAuthority106368.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority106345.actual selector witness) * (LeftAuthority106368.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106372

namespace LeftBound106380
def owner : Owner := ⟨.program ⟨214⟩, ⟨17799⟩⟩
def transferEvent : Nat := 106380
def frameStart : Nat := 106296
def rule : BoundRule := .sum [.predecessor 0 106378 .coefficient, .predecessor 1 106379 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106378 .coefficient)
      LeftAuthority106376.bound (LeftAuthority106376.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106377RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106376.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106376.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106379 .coefficient)
      LeftBound106372.bound (LeftBound106372.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106374RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106372.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106372.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority106376.bound, LeftBound106372.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106376.bound, LeftBound106372.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority106376.actual selector witness, LeftBound106372.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound106380

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
