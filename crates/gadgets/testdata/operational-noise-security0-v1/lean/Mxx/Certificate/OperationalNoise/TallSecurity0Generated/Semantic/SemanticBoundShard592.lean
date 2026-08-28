import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard083
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard084
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard538
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard541
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard591

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound86587
def owner : Owner := ⟨.program ⟨214⟩, ⟨15661⟩⟩
def transferEvent : Nat := 86587
def frameStart : Nat := 86514
def rule : BoundRule := .sum [.predecessor 0 86585 .coefficient, .predecessor 1 86586 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86585 .coefficient)
      LeftAuthority86583.bound (LeftAuthority86583.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events338.exact86584RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86583.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86583.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86586 .coefficient)
      LeftBound86579.bound (LeftBound86579.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events338.exact86581RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86579.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86579.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority86583.bound, LeftBound86579.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority86583.bound, LeftBound86579.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority86583.actual selector witness, LeftBound86579.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound86587

namespace LeftBound86591
def owner : Owner := ⟨.program ⟨214⟩, ⟨27216⟩⟩
def transferEvent : Nat := 86591
def frameStart : Nat := 86514
def rule : BoundRule := .product (.predecessor 0 86589 .coefficient) (.predecessor 1 86590 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86589 .coefficient)
      LeftBound86587.bound (LeftBound86587.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events338.exact86588RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86587.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86587.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86590 .coefficient)
      LeftAuthority86564.bound (LeftAuthority86564.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events338.exact86565RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86564.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86564.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound86587.bound LeftAuthority86564.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86587.bound, LeftAuthority86564.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound86587.actual selector witness) * (LeftAuthority86564.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound86591

namespace LeftBound86602
def owner : Owner := ⟨.program ⟨214⟩, ⟨15630⟩⟩
def transferEvent : Nat := 86602
def frameStart : Nat := 86514
def rule : BoundRule := .product (.predecessor 0 86600 .coefficient) (.predecessor 1 86601 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86600 .coefficient)
      LeftAuthority86575.bound (LeftAuthority86575.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events338.exact86576RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86575.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86575.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86601 .coefficient)
      LeftAuthority86598.bound (LeftAuthority86598.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events338.exact86599RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86598.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86598.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority86575.bound LeftAuthority86598.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority86575.bound, LeftAuthority86598.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority86575.actual selector witness) * (LeftAuthority86598.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound86602

namespace LeftBound86610
def owner : Owner := ⟨.program ⟨214⟩, ⟨15631⟩⟩
def transferEvent : Nat := 86610
def frameStart : Nat := 86514
def rule : BoundRule := .sum [.predecessor 0 86608 .coefficient, .predecessor 1 86609 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86608 .coefficient)
      LeftAuthority86606.bound (LeftAuthority86606.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events338.exact86607RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86606.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86606.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86609 .coefficient)
      LeftBound86602.bound (LeftBound86602.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events338.exact86604RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86602.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86602.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority86606.bound, LeftBound86602.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority86606.bound, LeftBound86602.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority86606.actual selector witness, LeftBound86602.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound86610

namespace LeftBound86614
def owner : Owner := ⟨.program ⟨214⟩, ⟨27220⟩⟩
def transferEvent : Nat := 86614
def frameStart : Nat := 86514
def rule : BoundRule := .sum [.predecessor 0 86612 .coefficient, .predecessor 1 86613 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86612 .coefficient)
      LeftBound86610.bound (LeftBound86610.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events338.exact86611RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86610.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86610.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86613 .coefficient)
      LeftBound86591.bound (LeftBound86591.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events338.exact86596RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86591.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86591.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound86610.bound, LeftBound86591.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86610.bound, LeftBound86591.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound86610.actual selector witness, LeftBound86591.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound86614

namespace LeftBound86627
def owner : Owner := ⟨.program ⟨214⟩, ⟨27218⟩⟩
def transferEvent : Nat := 86627
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 86625 .coefficient, .predecessor 1 86626 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86625 .coefficient)
      LeftBound86456.bound (LeftBound86456.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events338.exact86624RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86456.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86456.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86626 .coefficient)
      LeftBound86439.bound (LeftBound86439.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events337.exact86446RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86439.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86439.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound86456.bound, LeftBound86439.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86456.bound, LeftBound86439.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound86456.actual selector witness, LeftBound86439.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound86627

namespace LeftBound86630
def owner : Owner := ⟨.program ⟨214⟩, ⟨27218⟩⟩
def transferEvent : Nat := 86630
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 86624 .summary, .result 86446 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 86624 .summary)
      LeftBound86458.bound (LeftBound86458.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20971⟩⟩) (rawTerms := some (Proof.Events338.exact86624RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound86458.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 86446 .summary)
      LeftBound86441.bound (LeftBound86441.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27217⟩⟩) (rawTerms := some (Proof.Events337.exact86446RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound86441.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound86458.bound, LeftBound86441.bound]
def bound : CoeffClass := .finite ⟨1291978824159503986688, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86458.bound, LeftBound86441.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound86458.actual selector witness, LeftBound86441.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound86630

namespace LeftBound86654
def owner : Owner := ⟨.program ⟨214⟩, ⟨11134⟩⟩
def transferEvent : Nat := 86654
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 86652 .coefficient) (.predecessor 1 86653 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86652 .coefficient)
      LeftAuthority4149.bound (LeftAuthority4149.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events016.exact4150RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4149.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4149.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86653 .coefficient)
      LeftBound79918.bound (LeftBound79918.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79920RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79918.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79918.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority4149.bound LeftBound79918.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4149.bound, LeftBound79918.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority4149.actual selector witness) * (LeftBound79918.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound86654

namespace LeftBound86659
def owner : Owner := ⟨.program ⟨214⟩, ⟨7231⟩⟩
def transferEvent : Nat := 86659
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 86657 .coefficient) (.predecessor 1 86658 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86657 .coefficient)
      LeftBound79789.bound (LeftBound79789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events311.exact79790RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79789.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79789.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86658 .coefficient)
      LeftBound13485.bound (LeftBound13485.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13486RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13485.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13485.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound79789.bound LeftBound13485.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79789.bound, LeftBound13485.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound79789.actual selector witness) * (LeftBound13485.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound86659

namespace LeftBound86664
def owner : Owner := ⟨.program ⟨214⟩, ⟨11135⟩⟩
def transferEvent : Nat := 86664
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 86662 .coefficient, .predecessor 1 86663 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86662 .coefficient)
      LeftBound86659.bound (LeftBound86659.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events338.exact86661RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86659.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86659.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86663 .coefficient)
      LeftBound86654.bound (LeftBound86654.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events338.exact86656RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86654.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86654.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound86659.bound, LeftBound86654.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86659.bound, LeftBound86654.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound86659.actual selector witness, LeftBound86654.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound86664

namespace LeftBound86668
def owner : Owner := ⟨.program ⟨214⟩, ⟨11136⟩⟩
def transferEvent : Nat := 86668
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 86666 .coefficient, .predecessor 1 86667 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86666 .coefficient)
      LeftBound86664.bound (LeftBound86664.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events338.exact86665RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86664.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86664.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86667 .coefficient)
      LeftBound13477.bound (LeftBound13477.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13478RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13477.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13477.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound86664.bound, LeftBound13477.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86664.bound, LeftBound13477.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound86664.actual selector witness, LeftBound13477.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound86668

namespace LeftBound86669
def owner : Owner := ⟨.program ⟨214⟩, ⟨11136⟩⟩
def transferEvent : Nat := 86669
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨89⟩⟩]⟩ [⟨.result 13478 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13478 .coefficient)
      LeftBound13477.bound (LeftBound13477.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨89⟩⟩) (rawTerms := some (Proof.Events052.exact13478RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13477.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13477.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound13477.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13477.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound13477.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound86669

namespace LeftBound86674
def owner : Owner := ⟨.program ⟨214⟩, ⟨12166⟩⟩
def transferEvent : Nat := 86674
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 86672 .coefficient) (.predecessor 1 86673 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86672 .coefficient)
      LeftBound86668.bound (LeftBound86668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events338.exact86671RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86668.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86668.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86673 .coefficient)
      LeftAuthority4152.bound (LeftAuthority4152.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events016.exact4153RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4152.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4152.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound86668.bound LeftAuthority4152.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86668.bound, LeftAuthority4152.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound86668.actual selector witness) * (LeftAuthority4152.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound86674

namespace LeftBound86675
def owner : Owner := ⟨.program ⟨214⟩, ⟨12166⟩⟩
def transferEvent : Nat := 86675
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨12163⟩⟩], []⟩ [⟨.result 4153 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 4153 .coefficient)
      LeftAuthority4152.bound (LeftAuthority4152.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨12163⟩⟩) (rawTerms := some (Proof.Events016.exact4153RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4152.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4152.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority4152.bound []
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4152.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority4152.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound86675

namespace LeftBound86676
def owner : Owner := ⟨.program ⟨214⟩, ⟨12166⟩⟩
def transferEvent : Nat := 86676
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 86671 .summary) (.transfer 86675) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 86671 .summary)
      LeftBound86669.bound (LeftBound86669.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11136⟩⟩) (rawTerms := some (Proof.Events338.exact86671RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound86669.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 86675)
      LeftBound86675.bound (LeftBound86675.actual selector witness) := by
  exact .transfer (LeftBound86675.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound86669.bound LeftBound86675.bound
def bound : CoeffClass := .finite ⟨4992, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86669.bound, LeftBound86675.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound86669.actual selector witness) * (LeftBound86675.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound86676

namespace LeftBound86682
def owner : Owner := ⟨.program ⟨214⟩, ⟨12167⟩⟩
def transferEvent : Nat := 86682
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 86680 .coefficient) (.predecessor 1 86681 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86680 .coefficient)
      LeftAuthority4152.bound (LeftAuthority4152.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events016.exact4153RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4152.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4152.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86681 .coefficient)
      LeftBound79918.bound (LeftBound79918.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79920RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79918.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79918.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority4152.bound LeftBound79918.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4152.bound, LeftBound79918.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority4152.actual selector witness) * (LeftBound79918.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound86682

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
