import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard032
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard335
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard338
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard341

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound51099
def owner : Owner := ⟨.program ⟨214⟩, ⟨17058⟩⟩
def transferEvent : Nat := 51099
def frameStart : Nat := 51026
def rule : BoundRule := .sum [.predecessor 0 51097 .coefficient, .predecessor 1 51098 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51097 .coefficient)
      LeftAuthority51095.bound (LeftAuthority51095.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events199.exact51096RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51095.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51095.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51098 .coefficient)
      LeftBound51091.bound (LeftBound51091.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events199.exact51093RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51091.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51091.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority51095.bound, LeftBound51091.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority51095.bound, LeftBound51091.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority51095.actual selector witness, LeftBound51091.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound51099

namespace LeftBound51103
def owner : Owner := ⟨.program ⟨214⟩, ⟨30140⟩⟩
def transferEvent : Nat := 51103
def frameStart : Nat := 51026
def rule : BoundRule := .product (.predecessor 0 51101 .coefficient) (.predecessor 1 51102 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51101 .coefficient)
      LeftBound51099.bound (LeftBound51099.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events199.exact51100RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51099.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51099.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51102 .coefficient)
      LeftAuthority51076.bound (LeftAuthority51076.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events199.exact51077RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51076.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51076.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound51099.bound LeftAuthority51076.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51099.bound, LeftAuthority51076.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound51099.actual selector witness) * (LeftAuthority51076.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51103

namespace LeftBound51114
def owner : Owner := ⟨.program ⟨214⟩, ⟨18174⟩⟩
def transferEvent : Nat := 51114
def frameStart : Nat := 51026
def rule : BoundRule := .product (.predecessor 0 51112 .coefficient) (.predecessor 1 51113 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51112 .coefficient)
      LeftAuthority51087.bound (LeftAuthority51087.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events199.exact51088RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51087.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51087.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51113 .coefficient)
      LeftAuthority51110.bound (LeftAuthority51110.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events199.exact51111RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51110.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51110.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority51087.bound LeftAuthority51110.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority51087.bound, LeftAuthority51110.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority51087.actual selector witness) * (LeftAuthority51110.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51114

namespace LeftBound51122
def owner : Owner := ⟨.program ⟨214⟩, ⟨18175⟩⟩
def transferEvent : Nat := 51122
def frameStart : Nat := 51026
def rule : BoundRule := .sum [.predecessor 0 51120 .coefficient, .predecessor 1 51121 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51120 .coefficient)
      LeftAuthority51118.bound (LeftAuthority51118.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events199.exact51119RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51118.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51118.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51121 .coefficient)
      LeftBound51114.bound (LeftBound51114.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events199.exact51116RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51114.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51114.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority51118.bound, LeftBound51114.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority51118.bound, LeftBound51114.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority51118.actual selector witness, LeftBound51114.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound51122

namespace LeftBound51126
def owner : Owner := ⟨.program ⟨214⟩, ⟨30147⟩⟩
def transferEvent : Nat := 51126
def frameStart : Nat := 51026
def rule : BoundRule := .sum [.predecessor 0 51124 .coefficient, .predecessor 1 51125 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51124 .coefficient)
      LeftBound51122.bound (LeftBound51122.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events199.exact51123RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51122.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51122.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51125 .coefficient)
      LeftBound51103.bound (LeftBound51103.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events199.exact51108RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51103.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51103.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound51122.bound, LeftBound51103.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51122.bound, LeftBound51103.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound51122.actual selector witness, LeftBound51103.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound51126

namespace LeftBound51139
def owner : Owner := ⟨.program ⟨214⟩, ⟨30142⟩⟩
def transferEvent : Nat := 51139
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 51137 .coefficient, .predecessor 1 51138 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51137 .coefficient)
      LeftBound50968.bound (LeftBound50968.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events199.exact51136RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50968.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50968.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51138 .coefficient)
      LeftBound50951.bound (LeftBound50951.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events199.exact50958RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50951.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50951.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50968.bound, LeftBound50951.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50968.bound, LeftBound50951.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50968.actual selector witness, LeftBound50951.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound51139

namespace LeftBound51142
def owner : Owner := ⟨.program ⟨214⟩, ⟨30142⟩⟩
def transferEvent : Nat := 51142
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 51136 .summary, .result 50958 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 51136 .summary)
      LeftBound50970.bound (LeftBound50970.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22847⟩⟩) (rawTerms := some (Proof.Events199.exact51136RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50970.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50958 .summary)
      LeftBound50953.bound (LeftBound50953.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30141⟩⟩) (rawTerms := some (Proof.Events199.exact50958RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50953.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50970.bound, LeftBound50953.bound]
def bound : CoeffClass := .finite ⟨1292539135285018636288, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50970.bound, LeftBound50953.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50970.actual selector witness, LeftBound50953.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound51142

namespace LeftBound51166
def owner : Owner := ⟨.program ⟨214⟩, ⟨13165⟩⟩
def transferEvent : Nat := 51166
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 51164 .coefficient) (.predecessor 1 51165 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51164 .coefficient)
      LeftAuthority2360.bound (LeftAuthority2360.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events009.exact2361RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2360.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2360.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51165 .coefficient)
      LeftBound50668.bound (LeftBound50668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50670RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50668.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50668.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority2360.bound LeftBound50668.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2360.bound, LeftBound50668.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority2360.actual selector witness) * (LeftBound50668.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound51166

namespace LeftBound51171
def owner : Owner := ⟨.program ⟨214⟩, ⟨7283⟩⟩
def transferEvent : Nat := 51171
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 51169 .coefficient) (.predecessor 1 51170 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51169 .coefficient)
      LeftBound50539.bound (LeftBound50539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50539.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50539.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51170 .coefficient)
      LeftBound6972.bound (LeftBound6972.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact6973RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6972.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6972.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound50539.bound LeftBound6972.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50539.bound, LeftBound6972.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound50539.actual selector witness) * (LeftBound6972.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51171

namespace LeftBound51176
def owner : Owner := ⟨.program ⟨214⟩, ⟨13166⟩⟩
def transferEvent : Nat := 51176
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 51174 .coefficient, .predecessor 1 51175 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51174 .coefficient)
      LeftBound51171.bound (LeftBound51171.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events199.exact51173RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51171.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51171.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51175 .coefficient)
      LeftBound51166.bound (LeftBound51166.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events199.exact51168RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51166.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51166.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound51171.bound, LeftBound51166.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51171.bound, LeftBound51166.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound51171.actual selector witness, LeftBound51166.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound51176

namespace LeftBound51180
def owner : Owner := ⟨.program ⟨214⟩, ⟨13167⟩⟩
def transferEvent : Nat := 51180
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 51178 .coefficient, .predecessor 1 51179 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51178 .coefficient)
      LeftBound51176.bound (LeftBound51176.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events199.exact51177RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51176.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51176.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51179 .coefficient)
      LeftBound6964.bound (LeftBound6964.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact6965RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6964.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6964.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound51176.bound, LeftBound6964.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51176.bound, LeftBound6964.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound51176.actual selector witness, LeftBound6964.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound51180

namespace LeftBound51181
def owner : Owner := ⟨.program ⟨214⟩, ⟨13167⟩⟩
def transferEvent : Nat := 51181
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨103⟩⟩]⟩ [⟨.result 6965 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6965 .coefficient)
      LeftBound6964.bound (LeftBound6964.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨103⟩⟩) (rawTerms := some (Proof.Events027.exact6965RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6964.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6964.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound6964.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6964.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound6964.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound51181

namespace LeftBound51186
def owner : Owner := ⟨.program ⟨214⟩, ⟨13168⟩⟩
def transferEvent : Nat := 51186
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 51184 .coefficient) (.predecessor 1 51185 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51184 .coefficient)
      LeftBound51180.bound (LeftBound51180.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events199.exact51183RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51180.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51180.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51185 .coefficient)
      LeftAuthority2363.bound (LeftAuthority2363.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events009.exact2364RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2363.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2363.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound51180.bound LeftAuthority2363.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51180.bound, LeftAuthority2363.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound51180.actual selector witness) * (LeftAuthority2363.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51186

namespace LeftBound51187
def owner : Owner := ⟨.program ⟨214⟩, ⟨13168⟩⟩
def transferEvent : Nat := 51187
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨10245⟩⟩], []⟩ [⟨.result 2364 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2364 .coefficient)
      LeftAuthority2363.bound (LeftAuthority2363.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨10245⟩⟩) (rawTerms := some (Proof.Events009.exact2364RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2363.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2363.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority2363.bound []
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2363.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority2363.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound51187

namespace LeftBound51188
def owner : Owner := ⟨.program ⟨214⟩, ⟨13168⟩⟩
def transferEvent : Nat := 51188
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 51183 .summary) (.transfer 51187) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 51183 .summary)
      LeftBound51181.bound (LeftBound51181.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13167⟩⟩) (rawTerms := some (Proof.Events199.exact51183RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound51181.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 51187)
      LeftBound51187.bound (LeftBound51187.actual selector witness) := by
  exact .transfer (LeftBound51187.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound51181.bound LeftBound51187.bound
def bound : CoeffClass := .finite ⟨48256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51181.bound, LeftBound51187.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound51181.actual selector witness) * (LeftBound51187.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51188

namespace LeftBound51194
def owner : Owner := ⟨.program ⟨214⟩, ⟨10246⟩⟩
def transferEvent : Nat := 51194
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 51192 .coefficient) (.predecessor 1 51193 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51192 .coefficient)
      LeftAuthority2363.bound (LeftAuthority2363.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events009.exact2364RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2363.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2363.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51193 .coefficient)
      LeftBound50668.bound (LeftBound50668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50670RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50668.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50668.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority2363.bound LeftBound50668.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2363.bound, LeftBound50668.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority2363.actual selector witness) * (LeftBound50668.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound51194

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
