import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard489
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard527

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound78342
def owner : Owner := ⟨.program ⟨214⟩, ⟨27416⟩⟩
def transferEvent : Nat := 78342
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
end LeftBound78342

namespace LeftBound78343
def owner : Owner := ⟨.program ⟨214⟩, ⟨27416⟩⟩
def transferEvent : Nat := 78343
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 78338 .summary) (.transfer 78342) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 78338 .summary)
      LeftBound78337.bound (LeftBound78337.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27415⟩⟩) (rawTerms := some (Proof.Events306.exact78338RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound78337.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 78342)
      LeftBound78342.bound (LeftBound78342.actual selector witness) := by
  exact .transfer (LeftBound78342.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound78337.bound LeftBound78342.bound
def bound : CoeffClass := .finite ⟨4741665210358390854099402752, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound78337.bound, LeftBound78342.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound78337.actual selector witness) * (LeftBound78342.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound78343

namespace LeftBound78358
def owner : Owner := ⟨.program ⟨214⟩, ⟨27197⟩⟩
def transferEvent : Nat := 78358
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 78356 .coefficient) (.predecessor 1 78357 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78356 .coefficient)
      LeftBound71835.bound (LeftBound71835.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events280.exact71839RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71835.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71835.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78357 .coefficient)
      LeftAuthority78354.bound (LeftAuthority78354.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events306.exact78355RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78354.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78354.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound71835.bound LeftAuthority78354.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71835.bound, LeftAuthority78354.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound71835.actual selector witness) * (LeftAuthority78354.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound78358

namespace LeftBound78359
def owner : Owner := ⟨.program ⟨214⟩, ⟨27197⟩⟩
def transferEvent : Nat := 78359
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27195⟩⟩]⟩ [⟨.result 78355 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 78355 .coefficient)
      LeftAuthority78354.bound (LeftAuthority78354.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27195⟩⟩) (rawTerms := some (Proof.Events306.exact78355RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78354.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78354.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority78354.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority78354.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority78354.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound78359

namespace LeftBound78360
def owner : Owner := ⟨.program ⟨214⟩, ⟨27197⟩⟩
def transferEvent : Nat := 78360
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 71839 .summary) (.transfer 78359) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 71839 .summary)
      LeftBound71838.bound (LeftBound71838.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25832⟩⟩) (rawTerms := some (Proof.Events280.exact71839RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound71838.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 78359)
      LeftBound78359.bound (LeftBound78359.actual selector witness) := by
  exact .transfer (LeftBound78359.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound71838.bound LeftBound78359.bound
def bound : CoeffClass := .finite ⟨1291978822348200476672, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71838.bound, LeftBound78359.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound71838.actual selector witness) * (LeftBound78359.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound78360

namespace LeftBound78371
def owner : Owner := ⟨.program ⟨214⟩, ⟨20894⟩⟩
def transferEvent : Nat := 78371
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 78369 .coefficient) (.value (.predecessor 1 78370 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78369 .coefficient)
      LeftAuthority78367.bound (LeftAuthority78367.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events306.exact78368RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78367.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78367.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78370 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority78367.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority78367.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority78367.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound78371

namespace LeftBound78375
def owner : Owner := ⟨.program ⟨214⟩, ⟨20895⟩⟩
def transferEvent : Nat := 78375
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 78373 .coefficient) (.predecessor 1 78374 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78373 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78374 .coefficient)
      LeftBound78371.bound (LeftBound78371.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events306.exact78372RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78371.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78371.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound78371.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound78371.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound78371.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound78375

namespace LeftBound78376
def owner : Owner := ⟨.program ⟨214⟩, ⟨20895⟩⟩
def transferEvent : Nat := 78376
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20892⟩⟩]⟩ [⟨.result 78368 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 78368 .coefficient)
      LeftAuthority78367.bound (LeftAuthority78367.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20892⟩⟩) (rawTerms := some (Proof.Events306.exact78368RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78367.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78367.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority78367.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority78367.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority78367.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound78376

namespace LeftBound78377
def owner : Owner := ⟨.program ⟨214⟩, ⟨20895⟩⟩
def transferEvent : Nat := 78377
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 78376) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 78376)
      LeftBound78376.bound (LeftBound78376.actual selector witness) := by
  exact .transfer (LeftBound78376.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound78376.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound78376.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound78376.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound78377

namespace LeftBound78472
def owner : Owner := ⟨.program ⟨214⟩, ⟨15580⟩⟩
def transferEvent : Nat := 78472
def frameStart : Nat := 78433
def rule : BoundRule := .identity (.predecessor 0 78471 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78471 .coefficient)
      LeftAuthority78469.bound (LeftAuthority78469.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events306.exact78470RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78469.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78469.derived selector witness)

def rawBound : CoeffClass := LeftAuthority78469.bound
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority78469.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority78469.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound78472

namespace LeftBound78489
def owner : Owner := ⟨.program ⟨214⟩, ⟨15654⟩⟩
def transferEvent : Nat := 78489
def frameStart : Nat := 78433
def rule : BoundRule := .sum [.predecessor 0 78487 .coefficient, .predecessor 1 78488 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78487 .coefficient)
      LeftBound78472.bound (LeftBound78472.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound78472.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78488 .coefficient)
      LeftAuthority78485.bound (LeftAuthority78485.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority78485.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound78472.bound, LeftAuthority78485.bound]
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound78472.bound, LeftAuthority78485.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound78472.actual selector witness, LeftAuthority78485.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound78489

namespace LeftBound78492
def owner : Owner := ⟨.program ⟨214⟩, ⟨15655⟩⟩
def transferEvent : Nat := 78492
def frameStart : Nat := 78433
def rule : BoundRule := .identity (.predecessor 0 78491 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78491 .coefficient)
      LeftBound78489.bound (LeftBound78489.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound78489.derived selector witness)

def rawBound : CoeffClass := LeftBound78489.bound
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound78489.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound78489.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound78492

namespace LeftBound78498
def owner : Owner := ⟨.program ⟨214⟩, ⟨15656⟩⟩
def transferEvent : Nat := 78498
def frameStart : Nat := 78433
def rule : BoundRule := .product (.predecessor 0 78496 .coefficient) (.predecessor 1 78497 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78496 .coefficient)
      LeftAuthority78494.bound (LeftAuthority78494.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events306.exact78495RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78494.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78494.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78497 .coefficient)
      LeftBound78492.bound (LeftBound78492.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events306.exact78493RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78492.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78492.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority78494.bound LeftBound78492.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority78494.bound, LeftBound78492.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority78494.actual selector witness) * (LeftBound78492.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound78498

namespace LeftBound78506
def owner : Owner := ⟨.program ⟨214⟩, ⟨15657⟩⟩
def transferEvent : Nat := 78506
def frameStart : Nat := 78433
def rule : BoundRule := .sum [.predecessor 0 78504 .coefficient, .predecessor 1 78505 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78504 .coefficient)
      LeftAuthority78502.bound (LeftAuthority78502.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events306.exact78503RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78502.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78502.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78505 .coefficient)
      LeftBound78498.bound (LeftBound78498.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events306.exact78500RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78498.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78498.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority78502.bound, LeftBound78498.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority78502.bound, LeftBound78498.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority78502.actual selector witness, LeftBound78498.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound78506

namespace LeftBound78510
def owner : Owner := ⟨.program ⟨214⟩, ⟨27196⟩⟩
def transferEvent : Nat := 78510
def frameStart : Nat := 78433
def rule : BoundRule := .product (.predecessor 0 78508 .coefficient) (.predecessor 1 78509 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78508 .coefficient)
      LeftBound78506.bound (LeftBound78506.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events306.exact78507RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78506.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78506.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78509 .coefficient)
      LeftAuthority78483.bound (LeftAuthority78483.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events306.exact78484RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78483.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78483.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound78506.bound LeftAuthority78483.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound78506.bound, LeftAuthority78483.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound78506.actual selector witness) * (LeftAuthority78483.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound78510

namespace LeftBound78521
def owner : Owner := ⟨.program ⟨214⟩, ⟨17812⟩⟩
def transferEvent : Nat := 78521
def frameStart : Nat := 78433
def rule : BoundRule := .product (.predecessor 0 78519 .coefficient) (.predecessor 1 78520 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78519 .coefficient)
      LeftAuthority78494.bound (LeftAuthority78494.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events306.exact78495RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78494.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78494.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78520 .coefficient)
      LeftAuthority78517.bound (LeftAuthority78517.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events306.exact78518RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78517.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78517.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority78494.bound LeftAuthority78517.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority78494.bound, LeftAuthority78517.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority78494.actual selector witness) * (LeftAuthority78517.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound78521

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
