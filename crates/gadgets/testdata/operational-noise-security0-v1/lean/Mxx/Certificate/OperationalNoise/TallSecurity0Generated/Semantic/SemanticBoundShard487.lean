import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard080
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard437
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard440
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard486

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound71512
def owner : Owner := ⟨.program ⟨214⟩, ⟨27420⟩⟩
def transferEvent : Nat := 71512
def frameStart : Nat := 71435
def rule : BoundRule := .product (.predecessor 0 71510 .coefficient) (.predecessor 1 71511 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71510 .coefficient)
      LeftBound71508.bound (LeftBound71508.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events279.exact71509RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71508.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71508.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71511 .coefficient)
      LeftAuthority71485.bound (LeftAuthority71485.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events279.exact71486RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71485.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71485.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound71508.bound LeftAuthority71485.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71508.bound, LeftAuthority71485.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound71508.actual selector witness) * (LeftAuthority71485.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound71512

namespace LeftBound71523
def owner : Owner := ⟨.program ⟨214⟩, ⟨15746⟩⟩
def transferEvent : Nat := 71523
def frameStart : Nat := 71435
def rule : BoundRule := .product (.predecessor 0 71521 .coefficient) (.predecessor 1 71522 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71521 .coefficient)
      LeftAuthority71496.bound (LeftAuthority71496.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events279.exact71497RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71496.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71496.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71522 .coefficient)
      LeftAuthority71519.bound (LeftAuthority71519.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events279.exact71520RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71519.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71519.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority71496.bound LeftAuthority71519.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority71496.bound, LeftAuthority71519.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority71496.actual selector witness) * (LeftAuthority71519.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound71523

namespace LeftBound71531
def owner : Owner := ⟨.program ⟨214⟩, ⟨15747⟩⟩
def transferEvent : Nat := 71531
def frameStart : Nat := 71435
def rule : BoundRule := .sum [.predecessor 0 71529 .coefficient, .predecessor 1 71530 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71529 .coefficient)
      LeftAuthority71527.bound (LeftAuthority71527.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events279.exact71528RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71527.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71527.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71530 .coefficient)
      LeftBound71523.bound (LeftBound71523.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events279.exact71525RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71523.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71523.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority71527.bound, LeftBound71523.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority71527.bound, LeftBound71523.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority71527.actual selector witness, LeftBound71523.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71531

namespace LeftBound71535
def owner : Owner := ⟨.program ⟨214⟩, ⟨27424⟩⟩
def transferEvent : Nat := 71535
def frameStart : Nat := 71435
def rule : BoundRule := .sum [.predecessor 0 71533 .coefficient, .predecessor 1 71534 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71533 .coefficient)
      LeftBound71531.bound (LeftBound71531.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events279.exact71532RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71531.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71531.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71534 .coefficient)
      LeftBound71512.bound (LeftBound71512.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events279.exact71517RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71512.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71512.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound71531.bound, LeftBound71512.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71531.bound, LeftBound71512.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound71531.actual selector witness, LeftBound71512.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71535

namespace LeftBound71548
def owner : Owner := ⟨.program ⟨214⟩, ⟨27422⟩⟩
def transferEvent : Nat := 71548
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 71546 .coefficient, .predecessor 1 71547 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71546 .coefficient)
      LeftBound71377.bound (LeftBound71377.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events279.exact71545RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71377.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71377.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71547 .coefficient)
      LeftBound71360.bound (LeftBound71360.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events278.exact71367RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71360.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71360.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound71377.bound, LeftBound71360.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71377.bound, LeftBound71360.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound71377.actual selector witness, LeftBound71360.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71548

namespace LeftBound71551
def owner : Owner := ⟨.program ⟨214⟩, ⟨27422⟩⟩
def transferEvent : Nat := 71551
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 71545 .summary, .result 71367 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 71545 .summary)
      LeftBound71379.bound (LeftBound71379.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21111⟩⟩) (rawTerms := some (Proof.Events279.exact71545RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound71379.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 71367 .summary)
      LeftBound71362.bound (LeftBound71362.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27421⟩⟩) (rawTerms := some (Proof.Events278.exact71367RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound71362.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound71379.bound, LeftBound71362.bound]
def bound : CoeffClass := .finite ⟨1292001236604524572672, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71379.bound, LeftBound71362.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound71379.actual selector witness, LeftBound71362.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71551

namespace LeftBound71575
def owner : Owner := ⟨.program ⟨214⟩, ⟨11214⟩⟩
def transferEvent : Nat := 71575
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 71573 .coefficient) (.predecessor 1 71574 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71573 .coefficient)
      LeftAuthority3384.bound (LeftAuthority3384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events013.exact3385RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3384.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71574 .coefficient)
      LeftBound65293.bound (LeftBound65293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65293.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority3384.bound LeftBound65293.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3384.bound, LeftBound65293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority3384.actual selector witness) * (LeftBound65293.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound71575

namespace LeftBound71580
def owner : Owner := ⟨.program ⟨214⟩, ⟨7194⟩⟩
def transferEvent : Nat := 71580
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 71578 .coefficient) (.predecessor 1 71579 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71578 .coefficient)
      LeftBound65164.bound (LeftBound65164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65164.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71579 .coefficient)
      LeftBound12984.bound (LeftBound12984.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact12985RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12984.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12984.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound65164.bound LeftBound12984.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65164.bound, LeftBound12984.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound65164.actual selector witness) * (LeftBound12984.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound71580

namespace LeftBound71585
def owner : Owner := ⟨.program ⟨214⟩, ⟨11215⟩⟩
def transferEvent : Nat := 71585
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 71583 .coefficient, .predecessor 1 71584 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71583 .coefficient)
      LeftBound71580.bound (LeftBound71580.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events279.exact71582RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71580.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71580.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71584 .coefficient)
      LeftBound71575.bound (LeftBound71575.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events279.exact71577RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71575.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71575.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound71580.bound, LeftBound71575.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71580.bound, LeftBound71575.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound71580.actual selector witness, LeftBound71575.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71585

namespace LeftBound71589
def owner : Owner := ⟨.program ⟨214⟩, ⟨11216⟩⟩
def transferEvent : Nat := 71589
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 71587 .coefficient, .predecessor 1 71588 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71587 .coefficient)
      LeftBound71585.bound (LeftBound71585.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events279.exact71586RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71585.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71585.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71588 .coefficient)
      LeftBound12976.bound (LeftBound12976.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact12977RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12976.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12976.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound71585.bound, LeftBound12976.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71585.bound, LeftBound12976.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound71585.actual selector witness, LeftBound12976.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71589

namespace LeftBound71590
def owner : Owner := ⟨.program ⟨214⟩, ⟨11216⟩⟩
def transferEvent : Nat := 71590
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨90⟩⟩]⟩ [⟨.result 12977 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12977 .coefficient)
      LeftBound12976.bound (LeftBound12976.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨90⟩⟩) (rawTerms := some (Proof.Events050.exact12977RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12976.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12976.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound12976.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12976.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound12976.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound71590

namespace LeftBound71595
def owner : Owner := ⟨.program ⟨214⟩, ⟨13550⟩⟩
def transferEvent : Nat := 71595
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 71593 .coefficient) (.predecessor 1 71594 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71593 .coefficient)
      LeftBound71589.bound (LeftBound71589.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events279.exact71592RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71589.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71589.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71594 .coefficient)
      LeftAuthority3387.bound (LeftAuthority3387.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events013.exact3388RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3387.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3387.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound71589.bound LeftAuthority3387.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71589.bound, LeftAuthority3387.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound71589.actual selector witness) * (LeftAuthority3387.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound71595

namespace LeftBound71596
def owner : Owner := ⟨.program ⟨214⟩, ⟨13550⟩⟩
def transferEvent : Nat := 71596
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨13547⟩⟩], []⟩ [⟨.result 3388 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 3388 .coefficient)
      LeftAuthority3387.bound (LeftAuthority3387.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨13547⟩⟩) (rawTerms := some (Proof.Events013.exact3388RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3387.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3387.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority3387.bound []
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3387.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority3387.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound71596

namespace LeftBound71597
def owner : Owner := ⟨.program ⟨214⟩, ⟨13550⟩⟩
def transferEvent : Nat := 71597
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 71592 .summary) (.transfer 71596) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 71592 .summary)
      LeftBound71590.bound (LeftBound71590.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11216⟩⟩) (rawTerms := some (Proof.Events279.exact71592RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound71590.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 71596)
      LeftBound71596.bound (LeftBound71596.actual selector witness) := by
  exact .transfer (LeftBound71596.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound71590.bound LeftBound71596.bound
def bound : CoeffClass := .finite ⟨8320, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71590.bound, LeftBound71596.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound71590.actual selector witness) * (LeftBound71596.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound71597

namespace LeftBound71603
def owner : Owner := ⟨.program ⟨214⟩, ⟨13551⟩⟩
def transferEvent : Nat := 71603
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 71601 .coefficient) (.predecessor 1 71602 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71601 .coefficient)
      LeftAuthority3387.bound (LeftAuthority3387.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events013.exact3388RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3387.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3387.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71602 .coefficient)
      LeftBound65293.bound (LeftBound65293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65293.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority3387.bound LeftBound65293.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3387.bound, LeftBound65293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority3387.actual selector witness) * (LeftBound65293.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound71603

namespace LeftBound71608
def owner : Owner := ⟨.program ⟨214⟩, ⟨7211⟩⟩
def transferEvent : Nat := 71608
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 71606 .coefficient) (.predecessor 1 71607 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71606 .coefficient)
      LeftBound65164.bound (LeftBound65164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65164.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71607 .coefficient)
      LeftBound13025.bound (LeftBound13025.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact13026RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13025.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13025.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound65164.bound LeftBound13025.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65164.bound, LeftBound13025.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound65164.actual selector witness) * (LeftBound13025.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound71608

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
