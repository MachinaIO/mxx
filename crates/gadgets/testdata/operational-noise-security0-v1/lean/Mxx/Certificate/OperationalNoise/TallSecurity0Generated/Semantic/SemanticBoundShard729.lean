import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard693
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard728

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound106384
def owner : Owner := ⟨.program ⟨214⟩, ⟨27179⟩⟩
def transferEvent : Nat := 106384
def frameStart : Nat := 106296
def rule : BoundRule := .sum [.predecessor 0 106382 .coefficient, .predecessor 1 106383 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106382 .coefficient)
      LeftBound106380.bound (LeftBound106380.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106381RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106380.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106380.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106383 .coefficient)
      LeftBound106361.bound (LeftBound106361.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106366RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106361.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106361.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound106380.bound, LeftBound106361.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106380.bound, LeftBound106361.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound106380.actual selector witness, LeftBound106361.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound106384

namespace LeftBound106397
def owner : Owner := ⟨.program ⟨214⟩, ⟨27176⟩⟩
def transferEvent : Nat := 106397
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 106395 .coefficient, .predecessor 1 106396 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106395 .coefficient)
      LeftBound106250.bound (LeftBound106250.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106394RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106250.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106250.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106396 .coefficient)
      LeftBound106233.bound (LeftBound106233.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106240RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106233.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106233.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound106250.bound, LeftBound106233.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106250.bound, LeftBound106233.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound106250.actual selector witness, LeftBound106233.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound106397

namespace LeftBound106400
def owner : Owner := ⟨.program ⟨214⟩, ⟨27176⟩⟩
def transferEvent : Nat := 106400
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 106394 .summary, .result 106240 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 106394 .summary)
      LeftBound106252.bound (LeftBound106252.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20888⟩⟩) (rawTerms := some (Proof.Events415.exact106394RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound106252.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 106240 .summary)
      LeftBound106235.bound (LeftBound106235.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27175⟩⟩) (rawTerms := some (Proof.Events415.exact106240RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound106235.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound106252.bound, LeftBound106235.bound]
def bound : CoeffClass := .finite ⟨1291978824159503986688, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106252.bound, LeftBound106235.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound106252.actual selector witness, LeftBound106235.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound106400

namespace LeftBound106404
def owner : Owner := ⟨.program ⟨214⟩, ⟨27177⟩⟩
def transferEvent : Nat := 106404
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 106402 .coefficient) (.predecessor 1 106403 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106402 .coefficient)
      LeftBound106397.bound (LeftBound106397.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106401RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106397.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106397.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106403 .coefficient)
      LeftBound5778.bound (LeftBound5778.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5779RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5778.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5778.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound106397.bound LeftBound5778.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106397.bound, LeftBound5778.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound106397.actual selector witness) * (LeftBound5778.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106404

namespace LeftBound106405
def owner : Owner := ⟨.program ⟨214⟩, ⟨27177⟩⟩
def transferEvent : Nat := 106405
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩ [⟨.result 5775 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5775 .coefficient)
      LeftAuthority5774.bound (LeftAuthority5774.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6649⟩⟩) (rawTerms := some (Proof.Events022.exact5775RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5774.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5774.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5774.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5774.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5774.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound106405

namespace LeftBound106406
def owner : Owner := ⟨.program ⟨214⟩, ⟨27177⟩⟩
def transferEvent : Nat := 106406
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 106401 .summary) (.transfer 106405) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 106401 .summary)
      LeftBound106400.bound (LeftBound106400.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27176⟩⟩) (rawTerms := some (Proof.Events415.exact106401RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound106400.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 106405)
      LeftBound106405.bound (LeftBound106405.actual selector witness) := by
  exact .transfer (LeftBound106405.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound106400.bound LeftBound106405.bound
def bound : CoeffClass := .finite ⟨4741582956326566183208747008, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106400.bound, LeftBound106405.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound106400.actual selector witness) * (LeftBound106405.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106406

namespace LeftBound106421
def owner : Owner := ⟨.program ⟨214⟩, ⟨26958⟩⟩
def transferEvent : Nat := 106421
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 106419 .coefficient) (.predecessor 1 106420 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106419 .coefficient)
      LeftBound100696.bound (LeftBound100696.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100700RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100696.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100696.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106420 .coefficient)
      LeftAuthority106417.bound (LeftAuthority106417.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106418RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106417.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106417.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound100696.bound LeftAuthority106417.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100696.bound, LeftAuthority106417.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound100696.actual selector witness) * (LeftAuthority106417.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106421

namespace LeftBound106422
def owner : Owner := ⟨.program ⟨214⟩, ⟨26958⟩⟩
def transferEvent : Nat := 106422
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26956⟩⟩]⟩ [⟨.result 106418 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 106418 .coefficient)
      LeftAuthority106417.bound (LeftAuthority106417.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26956⟩⟩) (rawTerms := some (Proof.Events415.exact106418RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106417.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106417.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority106417.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106417.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority106417.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound106422

namespace LeftBound106423
def owner : Owner := ⟨.program ⟨214⟩, ⟨26958⟩⟩
def transferEvent : Nat := 106423
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 100700 .summary) (.transfer 106422) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 100700 .summary)
      LeftBound100699.bound (LeftBound100699.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25285⟩⟩) (rawTerms := some (Proof.Events393.exact100700RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound100699.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 106422)
      LeftBound106422.bound (LeftBound106422.actual selector witness) := by
  exact .transfer (LeftBound106422.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound100699.bound LeftBound106422.bound
def bound : CoeffClass := .finite ⟨1291933997458159304704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100699.bound, LeftBound106422.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound100699.actual selector witness) * (LeftBound106422.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106423

namespace LeftBound106434
def owner : Owner := ⟨.program ⟨214⟩, ⟨20743⟩⟩
def transferEvent : Nat := 106434
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 106432 .coefficient) (.value (.predecessor 1 106433 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106432 .coefficient)
      LeftAuthority106430.bound (LeftAuthority106430.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106431RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106430.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106430.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106433 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority106430.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106430.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority106430.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound106434

namespace LeftBound106438
def owner : Owner := ⟨.program ⟨214⟩, ⟨20744⟩⟩
def transferEvent : Nat := 106438
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 106436 .coefficient) (.predecessor 1 106437 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106436 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106437 .coefficient)
      LeftBound106434.bound (LeftBound106434.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106435RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106434.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106434.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound106434.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound106434.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound106434.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106438

namespace LeftBound106439
def owner : Owner := ⟨.program ⟨214⟩, ⟨20744⟩⟩
def transferEvent : Nat := 106439
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20741⟩⟩]⟩ [⟨.result 106431 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 106431 .coefficient)
      LeftAuthority106430.bound (LeftAuthority106430.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20741⟩⟩) (rawTerms := some (Proof.Events415.exact106431RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106430.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106430.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority106430.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106430.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority106430.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound106439

namespace LeftBound106440
def owner : Owner := ⟨.program ⟨214⟩, ⟨20744⟩⟩
def transferEvent : Nat := 106440
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 106439) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 106439)
      LeftBound106439.bound (LeftBound106439.actual selector witness) := by
  exact .transfer (LeftBound106439.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound106439.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound106439.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound106439.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106440

namespace LeftBound106511
def owner : Owner := ⟨.program ⟨214⟩, ⟨15413⟩⟩
def transferEvent : Nat := 106511
def frameStart : Nat := 106484
def rule : BoundRule := .identity (.predecessor 0 106510 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106510 .coefficient)
      LeftAuthority106508.bound (LeftAuthority106508.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events416.exact106509RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106508.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106508.derived selector witness)

def rawBound : CoeffClass := LeftAuthority106508.bound
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106508.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority106508.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound106511

namespace LeftBound106528
def owner : Owner := ⟨.program ⟨214⟩, ⟨15454⟩⟩
def transferEvent : Nat := 106528
def frameStart : Nat := 106484
def rule : BoundRule := .sum [.predecessor 0 106526 .coefficient, .predecessor 1 106527 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106526 .coefficient)
      LeftBound106511.bound (LeftBound106511.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound106511.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106527 .coefficient)
      LeftAuthority106524.bound (LeftAuthority106524.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority106524.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound106511.bound, LeftAuthority106524.bound]
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106511.bound, LeftAuthority106524.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound106511.actual selector witness, LeftAuthority106524.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound106528

namespace LeftBound106531
def owner : Owner := ⟨.program ⟨214⟩, ⟨15455⟩⟩
def transferEvent : Nat := 106531
def frameStart : Nat := 106484
def rule : BoundRule := .identity (.predecessor 0 106530 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106530 .coefficient)
      LeftBound106528.bound (LeftBound106528.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound106528.derived selector witness)

def rawBound : CoeffClass := LeftBound106528.bound
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106528.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound106528.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound106531

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
