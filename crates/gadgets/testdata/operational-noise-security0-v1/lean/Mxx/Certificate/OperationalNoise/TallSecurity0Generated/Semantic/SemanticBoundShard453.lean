import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard452

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound66963
def owner : Owner := ⟨.program ⟨214⟩, ⟨7874⟩⟩
def transferEvent : Nat := 66963
def frameStart : Nat := 66888
def rule : BoundRule := .scale (.predecessor 0 66961 .coefficient) (.value (.predecessor 1 66962 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66961 .coefficient)
      LeftAuthority66959.bound (LeftAuthority66959.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events261.exact66960RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66959.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66959.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66962 .coefficient)
      LeftAuthority66950.bound (LeftAuthority66950.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority66950.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority66959.bound LeftAuthority66950.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority66959.bound, LeftAuthority66950.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority66959.actual selector witness) * (LeftAuthority66950.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound66963

namespace LeftBound66966
def owner : Owner := ⟨.program ⟨214⟩, ⟨6767⟩⟩
def transferEvent : Nat := 66966
def frameStart : Nat := 66888
def rule : BoundRule := .identity (.predecessor 0 66965 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66965 .coefficient)
      LeftAuthority66953.bound (LeftAuthority66953.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events261.exact66954RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66953.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66953.derived selector witness)

def rawBound : CoeffClass := LeftAuthority66953.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority66953.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority66953.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound66966

namespace LeftBound66970
def owner : Owner := ⟨.program ⟨214⟩, ⟨7875⟩⟩
def transferEvent : Nat := 66970
def frameStart : Nat := 66888
def rule : BoundRule := .product (.predecessor 0 66968 .coefficient) (.predecessor 1 66969 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66968 .coefficient)
      LeftBound66966.bound (LeftBound66966.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events261.exact66967RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66966.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66966.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66969 .coefficient)
      LeftBound66963.bound (LeftBound66963.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events261.exact66964RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66963.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66963.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound66966.bound LeftBound66963.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66966.bound, LeftBound66963.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound66966.actual selector witness) * (LeftBound66963.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66970

namespace LeftBound66975
def owner : Owner := ⟨.program ⟨214⟩, ⟨12857⟩⟩
def transferEvent : Nat := 66975
def frameStart : Nat := 66888
def rule : BoundRule := .sum [.predecessor 0 66973 .coefficient, .predecessor 1 66974 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66973 .coefficient)
      LeftBound66970.bound (LeftBound66970.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events261.exact66972RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66970.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66970.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66974 .coefficient)
      LeftBound66947.bound (LeftBound66947.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events261.exact66949RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66947.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66947.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound66970.bound, LeftBound66947.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66970.bound, LeftBound66947.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound66970.actual selector witness, LeftBound66947.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound66975

namespace LeftBound66979
def owner : Owner := ⟨.program ⟨214⟩, ⟨25525⟩⟩
def transferEvent : Nat := 66979
def frameStart : Nat := 66888
def rule : BoundRule := .product (.predecessor 0 66977 .coefficient) (.predecessor 1 66978 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66977 .coefficient)
      LeftBound66975.bound (LeftBound66975.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events261.exact66976RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66975.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66975.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66978 .coefficient)
      LeftAuthority66932.bound (LeftAuthority66932.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events261.exact66933RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66932.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66932.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound66975.bound LeftAuthority66932.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66975.bound, LeftAuthority66932.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound66975.actual selector witness) * (LeftAuthority66932.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66979

namespace LeftBound66990
def owner : Owner := ⟨.program ⟨214⟩, ⟨16631⟩⟩
def transferEvent : Nat := 66990
def frameStart : Nat := 66888
def rule : BoundRule := .product (.predecessor 0 66988 .coefficient) (.predecessor 1 66989 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66988 .coefficient)
      LeftAuthority66943.bound (LeftAuthority66943.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events261.exact66944RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66943.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66943.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66989 .coefficient)
      LeftAuthority66986.bound (LeftAuthority66986.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events261.exact66987RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66986.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66986.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority66943.bound LeftAuthority66986.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority66943.bound, LeftAuthority66986.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority66943.actual selector witness) * (LeftAuthority66986.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66990

namespace LeftBound66998
def owner : Owner := ⟨.program ⟨214⟩, ⟨16632⟩⟩
def transferEvent : Nat := 66998
def frameStart : Nat := 66888
def rule : BoundRule := .sum [.predecessor 0 66996 .coefficient, .predecessor 1 66997 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66996 .coefficient)
      LeftAuthority66994.bound (LeftAuthority66994.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events261.exact66995RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66994.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66994.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66997 .coefficient)
      LeftBound66990.bound (LeftBound66990.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events261.exact66992RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66990.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66990.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority66994.bound, LeftBound66990.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority66994.bound, LeftBound66990.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority66994.actual selector witness, LeftBound66990.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound66998

namespace LeftBound67002
def owner : Owner := ⟨.program ⟨214⟩, ⟨25526⟩⟩
def transferEvent : Nat := 67002
def frameStart : Nat := 66888
def rule : BoundRule := .sum [.predecessor 0 67000 .coefficient, .predecessor 1 67001 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67000 .coefficient)
      LeftBound66998.bound (LeftBound66998.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events261.exact66999RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66998.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66998.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67001 .coefficient)
      LeftBound66979.bound (LeftBound66979.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events261.exact66984RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66979.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66979.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound66998.bound, LeftBound66979.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66998.bound, LeftBound66979.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound66998.actual selector witness, LeftBound66979.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound67002

namespace LeftBound67015
def owner : Owner := ⟨.program ⟨214⟩, ⟨25524⟩⟩
def transferEvent : Nat := 67015
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 67013 .coefficient, .predecessor 1 67014 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67013 .coefficient)
      LeftBound66836.bound (LeftBound66836.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events261.exact67012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66836.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66836.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67014 .coefficient)
      LeftBound66819.bound (LeftBound66819.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events261.exact66826RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66819.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66819.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound66836.bound, LeftBound66819.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66836.bound, LeftBound66819.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound66836.actual selector witness, LeftBound66819.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound67015

namespace LeftBound67018
def owner : Owner := ⟨.program ⟨214⟩, ⟨25524⟩⟩
def transferEvent : Nat := 67018
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 67012 .summary, .result 66826 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 67012 .summary)
      LeftBound66838.bound (LeftBound66838.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20031⟩⟩) (rawTerms := some (Proof.Events261.exact67012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound66838.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 66826 .summary)
      LeftBound66821.bound (LeftBound66821.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25523⟩⟩) (rawTerms := some (Proof.Events261.exact66826RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound66821.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound66838.bound, LeftBound66821.bound]
def bound : CoeffClass := .finite ⟨352146215809024, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66838.bound, LeftBound66821.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound66838.actual selector witness, LeftBound66821.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound67018

namespace LeftBound67022
def owner : Owner := ⟨.program ⟨214⟩, ⟨29374⟩⟩
def transferEvent : Nat := 67022
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 67020 .coefficient) (.predecessor 1 67021 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67020 .coefficient)
      LeftBound67015.bound (LeftBound67015.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events261.exact67019RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67015.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67015.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67021 .coefficient)
      LeftAuthority66741.bound (LeftAuthority66741.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events260.exact66742RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66741.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66741.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound67015.bound LeftAuthority66741.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67015.bound, LeftAuthority66741.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound67015.actual selector witness) * (LeftAuthority66741.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound67022

namespace LeftBound67023
def owner : Owner := ⟨.program ⟨214⟩, ⟨29374⟩⟩
def transferEvent : Nat := 67023
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29372⟩⟩]⟩ [⟨.result 66742 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 66742 .coefficient)
      LeftAuthority66741.bound (LeftAuthority66741.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29372⟩⟩) (rawTerms := some (Proof.Events260.exact66742RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66741.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66741.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority66741.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority66741.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority66741.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound67023

namespace LeftBound67024
def owner : Owner := ⟨.program ⟨214⟩, ⟨29374⟩⟩
def transferEvent : Nat := 67024
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 67019 .summary) (.transfer 67023) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 67019 .summary)
      LeftBound67018.bound (LeftBound67018.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25524⟩⟩) (rawTerms := some (Proof.Events261.exact67019RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound67018.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 67023)
      LeftBound67023.bound (LeftBound67023.actual selector witness) := by
  exact .transfer (LeftBound67023.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound67018.bound LeftBound67023.bound
def bound : CoeffClass := .finite ⟨1292382246358571024384, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67018.bound, LeftBound67023.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound67018.actual selector witness) * (LeftBound67023.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound67024

namespace LeftBound67035
def owner : Owner := ⟨.program ⟨214⟩, ⟨22406⟩⟩
def transferEvent : Nat := 67035
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 67033 .coefficient) (.value (.predecessor 1 67034 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67033 .coefficient)
      LeftAuthority67031.bound (LeftAuthority67031.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events261.exact67032RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67031.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67031.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67034 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority67031.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority67031.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority67031.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound67035

namespace LeftBound67039
def owner : Owner := ⟨.program ⟨214⟩, ⟨22407⟩⟩
def transferEvent : Nat := 67039
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 67037 .coefficient) (.predecessor 1 67038 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67037 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67038 .coefficient)
      LeftBound67035.bound (LeftBound67035.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events261.exact67036RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67035.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67035.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound67035.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound67035.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound67035.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound67039

namespace LeftBound67040
def owner : Owner := ⟨.program ⟨214⟩, ⟨22407⟩⟩
def transferEvent : Nat := 67040
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22404⟩⟩]⟩ [⟨.result 67032 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 67032 .coefficient)
      LeftAuthority67031.bound (LeftAuthority67031.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22404⟩⟩) (rawTerms := some (Proof.Events261.exact67032RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67031.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67031.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority67031.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority67031.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority67031.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound67040

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
