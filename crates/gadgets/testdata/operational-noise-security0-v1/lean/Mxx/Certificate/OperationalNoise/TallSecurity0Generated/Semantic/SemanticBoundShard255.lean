import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard048
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard234
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard237
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard254

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound38394
def owner : Owner := ⟨.program ⟨214⟩, ⟨16599⟩⟩
def transferEvent : Nat := 38394
def frameStart : Nat := 38329
def rule : BoundRule := .product (.predecessor 0 38392 .coefficient) (.predecessor 1 38393 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38392 .coefficient)
      LeftAuthority38390.bound (LeftAuthority38390.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events149.exact38391RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38390.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38390.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38393 .coefficient)
      LeftBound38388.bound (LeftBound38388.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events149.exact38389RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38388.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38388.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority38390.bound LeftBound38388.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority38390.bound, LeftBound38388.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority38390.actual selector witness) * (LeftBound38388.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38394

namespace LeftBound38402
def owner : Owner := ⟨.program ⟨214⟩, ⟨16600⟩⟩
def transferEvent : Nat := 38402
def frameStart : Nat := 38329
def rule : BoundRule := .sum [.predecessor 0 38400 .coefficient, .predecessor 1 38401 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38400 .coefficient)
      LeftAuthority38398.bound (LeftAuthority38398.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events149.exact38399RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38398.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38398.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38401 .coefficient)
      LeftBound38394.bound (LeftBound38394.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events149.exact38396RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38394.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38394.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority38398.bound, LeftBound38394.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority38398.bound, LeftBound38394.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority38398.actual selector witness, LeftBound38394.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound38402

namespace LeftBound38406
def owner : Owner := ⟨.program ⟨214⟩, ⟨29195⟩⟩
def transferEvent : Nat := 38406
def frameStart : Nat := 38329
def rule : BoundRule := .product (.predecessor 0 38404 .coefficient) (.predecessor 1 38405 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38404 .coefficient)
      LeftBound38402.bound (LeftBound38402.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events150.exact38403RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38402.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38402.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38405 .coefficient)
      LeftAuthority38379.bound (LeftAuthority38379.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events149.exact38380RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38379.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38379.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound38402.bound LeftAuthority38379.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38402.bound, LeftAuthority38379.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound38402.actual selector witness) * (LeftAuthority38379.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38406

namespace LeftBound38417
def owner : Owner := ⟨.program ⟨214⟩, ⟨18212⟩⟩
def transferEvent : Nat := 38417
def frameStart : Nat := 38329
def rule : BoundRule := .product (.predecessor 0 38415 .coefficient) (.predecessor 1 38416 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38415 .coefficient)
      LeftAuthority38390.bound (LeftAuthority38390.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events149.exact38391RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38390.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38390.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38416 .coefficient)
      LeftAuthority38413.bound (LeftAuthority38413.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events150.exact38414RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38413.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38413.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority38390.bound LeftAuthority38413.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority38390.bound, LeftAuthority38413.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority38390.actual selector witness) * (LeftAuthority38413.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38417

namespace LeftBound38425
def owner : Owner := ⟨.program ⟨214⟩, ⟨18213⟩⟩
def transferEvent : Nat := 38425
def frameStart : Nat := 38329
def rule : BoundRule := .sum [.predecessor 0 38423 .coefficient, .predecessor 1 38424 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38423 .coefficient)
      LeftAuthority38421.bound (LeftAuthority38421.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events150.exact38422RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38421.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38421.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38424 .coefficient)
      LeftBound38417.bound (LeftBound38417.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events150.exact38419RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38417.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38417.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority38421.bound, LeftBound38417.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority38421.bound, LeftBound38417.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority38421.actual selector witness, LeftBound38417.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound38425

namespace LeftBound38429
def owner : Owner := ⟨.program ⟨214⟩, ⟨29199⟩⟩
def transferEvent : Nat := 38429
def frameStart : Nat := 38329
def rule : BoundRule := .sum [.predecessor 0 38427 .coefficient, .predecessor 1 38428 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38427 .coefficient)
      LeftBound38425.bound (LeftBound38425.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events150.exact38426RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38425.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38425.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38428 .coefficient)
      LeftBound38406.bound (LeftBound38406.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events150.exact38411RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38406.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38406.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound38425.bound, LeftBound38406.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38425.bound, LeftBound38406.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound38425.actual selector witness, LeftBound38406.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound38429

namespace LeftBound38442
def owner : Owner := ⟨.program ⟨214⟩, ⟨29197⟩⟩
def transferEvent : Nat := 38442
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 38440 .coefficient, .predecessor 1 38441 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38440 .coefficient)
      LeftBound38271.bound (LeftBound38271.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events150.exact38439RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38271.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38271.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38441 .coefficient)
      LeftBound38254.bound (LeftBound38254.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events149.exact38261RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38254.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38254.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound38271.bound, LeftBound38254.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38271.bound, LeftBound38254.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound38271.actual selector witness, LeftBound38254.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound38442

namespace LeftBound38445
def owner : Owner := ⟨.program ⟨214⟩, ⟨29197⟩⟩
def transferEvent : Nat := 38445
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 38439 .summary, .result 38261 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 38439 .summary)
      LeftBound38273.bound (LeftBound38273.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22275⟩⟩) (rawTerms := some (Proof.Events150.exact38439RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound38273.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 38261 .summary)
      LeftBound38256.bound (LeftBound38256.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29196⟩⟩) (rawTerms := some (Proof.Events149.exact38261RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound38256.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound38273.bound, LeftBound38256.bound]
def bound : CoeffClass := .finite ⟨1292337423279833362432, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38273.bound, LeftBound38256.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound38273.actual selector witness, LeftBound38256.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound38445

namespace LeftBound38469
def owner : Owner := ⟨.program ⟨214⟩, ⟨12389⟩⟩
def transferEvent : Nat := 38469
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 38467 .coefficient) (.predecessor 1 38468 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38467 .coefficient)
      LeftAuthority1704.bound (LeftAuthority1704.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events006.exact1705RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1704.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1704.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38468 .coefficient)
      LeftBound36043.bound (LeftBound36043.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact36045RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36043.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36043.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority1704.bound LeftBound36043.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1704.bound, LeftBound36043.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority1704.actual selector witness) * (LeftBound36043.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound38469

namespace LeftBound38474
def owner : Owner := ⟨.program ⟨214⟩, ⟨7317⟩⟩
def transferEvent : Nat := 38474
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 38472 .coefficient) (.predecessor 1 38473 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38472 .coefficient)
      LeftBound35914.bound (LeftBound35914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact35915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35914.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35914.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38473 .coefficient)
      LeftBound8976.bound (LeftBound8976.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact8977RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8976.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8976.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound35914.bound LeftBound8976.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35914.bound, LeftBound8976.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound35914.actual selector witness) * (LeftBound8976.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38474

namespace LeftBound38479
def owner : Owner := ⟨.program ⟨214⟩, ⟨12390⟩⟩
def transferEvent : Nat := 38479
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 38477 .coefficient, .predecessor 1 38478 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38477 .coefficient)
      LeftBound38474.bound (LeftBound38474.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events150.exact38476RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38474.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38474.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38478 .coefficient)
      LeftBound38469.bound (LeftBound38469.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events150.exact38471RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38469.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38469.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound38474.bound, LeftBound38469.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38474.bound, LeftBound38469.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound38474.actual selector witness, LeftBound38469.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound38479

namespace LeftBound38483
def owner : Owner := ⟨.program ⟨214⟩, ⟨12391⟩⟩
def transferEvent : Nat := 38483
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 38481 .coefficient, .predecessor 1 38482 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38481 .coefficient)
      LeftBound38479.bound (LeftBound38479.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events150.exact38480RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38479.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38479.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38482 .coefficient)
      LeftBound8968.bound (LeftBound8968.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact8969RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8968.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8968.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound38479.bound, LeftBound8968.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38479.bound, LeftBound8968.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound38479.actual selector witness, LeftBound8968.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound38483

namespace LeftBound38484
def owner : Owner := ⟨.program ⟨214⟩, ⟨12391⟩⟩
def transferEvent : Nat := 38484
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
end LeftBound38484

namespace LeftBound38489
def owner : Owner := ⟨.program ⟨214⟩, ⟨12392⟩⟩
def transferEvent : Nat := 38489
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 38487 .coefficient) (.predecessor 1 38488 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38487 .coefficient)
      LeftBound38483.bound (LeftBound38483.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events150.exact38486RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38483.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38483.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38488 .coefficient)
      LeftAuthority1707.bound (LeftAuthority1707.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events006.exact1708RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1707.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1707.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound38483.bound LeftAuthority1707.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38483.bound, LeftAuthority1707.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound38483.actual selector witness) * (LeftAuthority1707.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38489

namespace LeftBound38490
def owner : Owner := ⟨.program ⟨214⟩, ⟨12392⟩⟩
def transferEvent : Nat := 38490
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨9830⟩⟩], []⟩ [⟨.result 1708 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 1708 .coefficient)
      LeftAuthority1707.bound (LeftAuthority1707.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨9830⟩⟩) (rawTerms := some (Proof.Events006.exact1708RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1707.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1707.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority1707.bound []
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1707.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority1707.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound38490

namespace LeftBound38491
def owner : Owner := ⟨.program ⟨214⟩, ⟨12392⟩⟩
def transferEvent : Nat := 38491
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 38486 .summary) (.transfer 38490) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 38486 .summary)
      LeftBound38484.bound (LeftBound38484.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12391⟩⟩) (rawTerms := some (Proof.Events150.exact38486RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound38484.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 38490)
      LeftBound38490.bound (LeftBound38490.actual selector witness) := by
  exact .transfer (LeftBound38490.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound38484.bound LeftBound38490.bound
def bound : CoeffClass := .finite ⟨33280, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38484.bound, LeftBound38490.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound38484.actual selector witness) * (LeftBound38490.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38491

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
