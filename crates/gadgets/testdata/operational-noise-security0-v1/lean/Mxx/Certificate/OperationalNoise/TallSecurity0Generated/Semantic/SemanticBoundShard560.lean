import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard048
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard049
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard538
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard541
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard559

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound82348
def owner : Owner := ⟨.program ⟨214⟩, ⟨12375⟩⟩
def transferEvent : Nat := 82348
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 82346 .coefficient, .predecessor 1 82347 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82346 .coefficient)
      LeftBound82344.bound (LeftBound82344.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82345RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82344.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82344.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82347 .coefficient)
      LeftBound8968.bound (LeftBound8968.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact8969RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8968.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8968.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound82344.bound, LeftBound8968.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82344.bound, LeftBound8968.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound82344.actual selector witness, LeftBound8968.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound82348

namespace LeftBound82349
def owner : Owner := ⟨.program ⟨214⟩, ⟨12375⟩⟩
def transferEvent : Nat := 82349
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨99⟩⟩]⟩ [⟨.result 8969 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 8969 .coefficient)
      LeftBound8968.bound (LeftBound8968.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨99⟩⟩) (rawTerms := some (Proof.Events035.exact8969RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8968.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8968.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound8968.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8968.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound8968.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound82349

namespace LeftBound82354
def owner : Owner := ⟨.program ⟨214⟩, ⟨12376⟩⟩
def transferEvent : Nat := 82354
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 82352 .coefficient) (.predecessor 1 82353 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82352 .coefficient)
      LeftBound82348.bound (LeftBound82348.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82351RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82348.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82348.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82353 .coefficient)
      LeftAuthority3945.bound (LeftAuthority3945.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events015.exact3946RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3945.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3945.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound82348.bound LeftAuthority3945.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82348.bound, LeftAuthority3945.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound82348.actual selector witness) * (LeftAuthority3945.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82354

namespace LeftBound82355
def owner : Owner := ⟨.program ⟨214⟩, ⟨12376⟩⟩
def transferEvent : Nat := 82355
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨9820⟩⟩], []⟩ [⟨.result 3946 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 3946 .coefficient)
      LeftAuthority3945.bound (LeftAuthority3945.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨9820⟩⟩) (rawTerms := some (Proof.Events015.exact3946RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3945.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3945.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority3945.bound []
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3945.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority3945.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound82355

namespace LeftBound82356
def owner : Owner := ⟨.program ⟨214⟩, ⟨12376⟩⟩
def transferEvent : Nat := 82356
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 82351 .summary) (.transfer 82355) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 82351 .summary)
      LeftBound82349.bound (LeftBound82349.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12375⟩⟩) (rawTerms := some (Proof.Events321.exact82351RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound82349.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 82355)
      LeftBound82355.bound (LeftBound82355.actual selector witness) := by
  exact .transfer (LeftBound82355.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound82349.bound LeftBound82355.bound
def bound : CoeffClass := .finite ⟨33280, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82349.bound, LeftBound82355.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound82349.actual selector witness) * (LeftBound82355.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82356

namespace LeftBound82362
def owner : Owner := ⟨.program ⟨214⟩, ⟨9821⟩⟩
def transferEvent : Nat := 82362
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 82360 .coefficient) (.predecessor 1 82361 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82360 .coefficient)
      LeftAuthority3945.bound (LeftAuthority3945.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events015.exact3946RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3945.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3945.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82361 .coefficient)
      LeftBound79918.bound (LeftBound79918.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79920RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79918.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79918.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority3945.bound LeftBound79918.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3945.bound, LeftBound79918.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority3945.actual selector witness) * (LeftBound79918.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound82362

namespace LeftBound82367
def owner : Owner := ⟨.program ⟨214⟩, ⟨7221⟩⟩
def transferEvent : Nat := 82367
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 82365 .coefficient) (.predecessor 1 82366 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82365 .coefficient)
      LeftBound79789.bound (LeftBound79789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events311.exact79790RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79789.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79789.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82366 .coefficient)
      LeftBound9017.bound (LeftBound9017.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9018RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9017.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9017.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound79789.bound LeftBound9017.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79789.bound, LeftBound9017.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound79789.actual selector witness) * (LeftBound9017.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82367

namespace LeftBound82372
def owner : Owner := ⟨.program ⟨214⟩, ⟨9822⟩⟩
def transferEvent : Nat := 82372
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 82370 .coefficient, .predecessor 1 82371 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82370 .coefficient)
      LeftBound82367.bound (LeftBound82367.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82369RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82367.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82367.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82371 .coefficient)
      LeftBound82362.bound (LeftBound82362.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82364RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82362.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82362.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound82367.bound, LeftBound82362.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82367.bound, LeftBound82362.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound82367.actual selector witness, LeftBound82362.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound82372

namespace LeftBound82376
def owner : Owner := ⟨.program ⟨214⟩, ⟨9823⟩⟩
def transferEvent : Nat := 82376
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 82374 .coefficient, .predecessor 1 82375 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82374 .coefficient)
      LeftBound82372.bound (LeftBound82372.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82373RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82372.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82372.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82375 .coefficient)
      LeftBound9009.bound (LeftBound9009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9010RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9009.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound82372.bound, LeftBound9009.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82372.bound, LeftBound9009.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound82372.actual selector witness, LeftBound9009.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound82376

namespace LeftBound82377
def owner : Owner := ⟨.program ⟨214⟩, ⟨9823⟩⟩
def transferEvent : Nat := 82377
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨79⟩⟩]⟩ [⟨.result 9010 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9010 .coefficient)
      LeftBound9009.bound (LeftBound9009.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨79⟩⟩) (rawTerms := some (Proof.Events035.exact9010RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9009.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound9009.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9009.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound9009.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound82377

namespace LeftBound82382
def owner : Owner := ⟨.program ⟨214⟩, ⟨9824⟩⟩
def transferEvent : Nat := 82382
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 82380 .coefficient) (.predecessor 1 82381 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82380 .coefficient)
      LeftBound82376.bound (LeftBound82376.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82379RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82376.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82376.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82381 .coefficient)
      LeftBound9006.bound (LeftBound9006.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9007RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9006.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9006.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound82376.bound LeftBound9006.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82376.bound, LeftBound9006.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound82376.actual selector witness) * (LeftBound9006.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82382

namespace LeftBound82383
def owner : Owner := ⟨.program ⟨214⟩, ⟨9824⟩⟩
def transferEvent : Nat := 82383
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7867⟩⟩]⟩ [⟨.result 9003 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9003 .coefficient)
      LeftAuthority9002.bound (LeftAuthority9002.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7867⟩⟩) (rawTerms := some (Proof.Events035.exact9003RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9002.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9002.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority9002.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9002.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority9002.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound82383

namespace LeftBound82384
def owner : Owner := ⟨.program ⟨214⟩, ⟨9824⟩⟩
def transferEvent : Nat := 82384
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 82379 .summary) (.transfer 82383) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 82379 .summary)
      LeftBound82377.bound (LeftBound82377.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9823⟩⟩) (rawTerms := some (Proof.Events321.exact82379RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound82377.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 82383)
      LeftBound82383.bound (LeftBound82383.actual selector witness) := by
  exact .transfer (LeftBound82383.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound82377.bound LeftBound82383.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82377.bound, LeftBound82383.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound82377.actual selector witness) * (LeftBound82383.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82384

namespace LeftBound82392
def owner : Owner := ⟨.program ⟨214⟩, ⟨12377⟩⟩
def transferEvent : Nat := 82392
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 82390 .coefficient, .predecessor 1 82391 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82390 .coefficient)
      LeftBound82382.bound (LeftBound82382.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82389RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82382.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82382.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82391 .coefficient)
      LeftBound82354.bound (LeftBound82354.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82359RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82354.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82354.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound82382.bound, LeftBound82354.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82382.bound, LeftBound82354.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound82382.actual selector witness, LeftBound82354.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound82392

namespace LeftBound82394
def owner : Owner := ⟨.program ⟨214⟩, ⟨12377⟩⟩
def transferEvent : Nat := 82394
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 82389 .summary, .result 82359 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 82389 .summary)
      LeftBound82384.bound (LeftBound82384.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9824⟩⟩) (rawTerms := some (Proof.Events321.exact82389RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound82384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 82359 .summary)
      LeftBound82356.bound (LeftBound82356.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12376⟩⟩) (rawTerms := some (Proof.Events321.exact82359RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound82356.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound82384.bound, LeftBound82356.bound]
def bound : CoeffClass := .finite ⟨95453696, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82384.bound, LeftBound82356.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound82384.actual selector witness, LeftBound82356.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound82394

namespace LeftBound82398
def owner : Owner := ⟨.program ⟨214⟩, ⟨25374⟩⟩
def transferEvent : Nat := 82398
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 82396 .coefficient) (.predecessor 1 82397 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82396 .coefficient)
      LeftBound82392.bound (LeftBound82392.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82395RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82392.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82392.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82397 .coefficient)
      LeftAuthority82330.bound (LeftAuthority82330.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82331RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82330.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82330.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound82392.bound LeftAuthority82330.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82392.bound, LeftAuthority82330.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound82392.actual selector witness) * (LeftAuthority82330.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82398

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
