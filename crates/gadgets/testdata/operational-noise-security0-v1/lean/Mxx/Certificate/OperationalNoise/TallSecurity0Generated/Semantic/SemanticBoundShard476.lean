import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard068
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard437
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard440
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard475

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound70054
def owner : Owner := ⟨.program ⟨214⟩, ⟨16132⟩⟩
def transferEvent : Nat := 70054
def frameStart : Nat := 69989
def rule : BoundRule := .product (.predecessor 0 70052 .coefficient) (.predecessor 1 70053 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70052 .coefficient)
      LeftAuthority70050.bound (LeftAuthority70050.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact70051RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70050.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70050.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70053 .coefficient)
      LeftBound70048.bound (LeftBound70048.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact70049RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70048.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70048.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority70050.bound LeftBound70048.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority70050.bound, LeftBound70048.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority70050.actual selector witness) * (LeftBound70048.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70054

namespace LeftBound70062
def owner : Owner := ⟨.program ⟨214⟩, ⟨16133⟩⟩
def transferEvent : Nat := 70062
def frameStart : Nat := 69989
def rule : BoundRule := .sum [.predecessor 0 70060 .coefficient, .predecessor 1 70061 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70060 .coefficient)
      LeftAuthority70058.bound (LeftAuthority70058.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact70059RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70058.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70058.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70061 .coefficient)
      LeftBound70054.bound (LeftBound70054.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact70056RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70054.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70054.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority70058.bound, LeftBound70054.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority70058.bound, LeftBound70054.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority70058.actual selector witness, LeftBound70054.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound70062

namespace LeftBound70066
def owner : Owner := ⟨.program ⟨214⟩, ⟨28071⟩⟩
def transferEvent : Nat := 70066
def frameStart : Nat := 69989
def rule : BoundRule := .product (.predecessor 0 70064 .coefficient) (.predecessor 1 70065 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70064 .coefficient)
      LeftBound70062.bound (LeftBound70062.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact70063RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70062.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70062.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70065 .coefficient)
      LeftAuthority70039.bound (LeftAuthority70039.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact70040RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70039.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70039.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound70062.bound LeftAuthority70039.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70062.bound, LeftAuthority70039.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound70062.actual selector witness) * (LeftAuthority70039.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70066

namespace LeftBound70077
def owner : Owner := ⟨.program ⟨214⟩, ⟨16103⟩⟩
def transferEvent : Nat := 70077
def frameStart : Nat := 69989
def rule : BoundRule := .product (.predecessor 0 70075 .coefficient) (.predecessor 1 70076 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70075 .coefficient)
      LeftAuthority70050.bound (LeftAuthority70050.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact70051RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70050.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70050.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70076 .coefficient)
      LeftAuthority70073.bound (LeftAuthority70073.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact70074RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70073.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70073.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority70050.bound LeftAuthority70073.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority70050.bound, LeftAuthority70073.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority70050.actual selector witness) * (LeftAuthority70073.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70077

namespace LeftBound70085
def owner : Owner := ⟨.program ⟨214⟩, ⟨16104⟩⟩
def transferEvent : Nat := 70085
def frameStart : Nat := 69989
def rule : BoundRule := .sum [.predecessor 0 70083 .coefficient, .predecessor 1 70084 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70083 .coefficient)
      LeftAuthority70081.bound (LeftAuthority70081.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact70082RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70081.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70081.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70084 .coefficient)
      LeftBound70077.bound (LeftBound70077.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact70079RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70077.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70077.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority70081.bound, LeftBound70077.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority70081.bound, LeftBound70077.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority70081.actual selector witness, LeftBound70077.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound70085

namespace LeftBound70089
def owner : Owner := ⟨.program ⟨214⟩, ⟨28075⟩⟩
def transferEvent : Nat := 70089
def frameStart : Nat := 69989
def rule : BoundRule := .sum [.predecessor 0 70087 .coefficient, .predecessor 1 70088 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70087 .coefficient)
      LeftBound70085.bound (LeftBound70085.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact70086RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70085.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70085.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70088 .coefficient)
      LeftBound70066.bound (LeftBound70066.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact70071RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70066.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70066.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound70085.bound, LeftBound70066.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70085.bound, LeftBound70066.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound70085.actual selector witness, LeftBound70066.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound70089

namespace LeftBound70102
def owner : Owner := ⟨.program ⟨214⟩, ⟨28073⟩⟩
def transferEvent : Nat := 70102
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 70100 .coefficient, .predecessor 1 70101 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70100 .coefficient)
      LeftBound69931.bound (LeftBound69931.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact70099RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69931.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69931.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70101 .coefficient)
      LeftBound69914.bound (LeftBound69914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact69921RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69914.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69914.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69931.bound, LeftBound69914.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69931.bound, LeftBound69914.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound69931.actual selector witness, LeftBound69914.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound70102

namespace LeftBound70105
def owner : Owner := ⟨.program ⟨214⟩, ⟨28073⟩⟩
def transferEvent : Nat := 70105
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 70099 .summary, .result 69921 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 70099 .summary)
      LeftBound69933.bound (LeftBound69933.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21543⟩⟩) (rawTerms := some (Proof.Events273.exact70099RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound69933.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 69921 .summary)
      LeftBound69916.bound (LeftBound69916.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28072⟩⟩) (rawTerms := some (Proof.Events273.exact69921RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound69916.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69933.bound, LeftBound69916.bound]
def bound : CoeffClass := .finite ⟨1292113298829627502592, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69933.bound, LeftBound69916.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound69933.actual selector witness, LeftBound69916.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound70105

namespace LeftBound70129
def owner : Owner := ⟨.program ⟨214⟩, ⟨11466⟩⟩
def transferEvent : Nat := 70129
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 70127 .coefficient) (.predecessor 1 70128 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70127 .coefficient)
      LeftAuthority3315.bound (LeftAuthority3315.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events012.exact3316RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3315.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3315.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70128 .coefficient)
      LeftBound65293.bound (LeftBound65293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65293.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority3315.bound LeftBound65293.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3315.bound, LeftBound65293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority3315.actual selector witness) * (LeftBound65293.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound70129

namespace LeftBound70134
def owner : Owner := ⟨.program ⟨214⟩, ⟨7197⟩⟩
def transferEvent : Nat := 70134
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 70132 .coefficient) (.predecessor 1 70133 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70132 .coefficient)
      LeftBound65164.bound (LeftBound65164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65164.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70133 .coefficient)
      LeftBound11481.bound (LeftBound11481.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11482RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11481.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11481.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound65164.bound LeftBound11481.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65164.bound, LeftBound11481.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound65164.actual selector witness) * (LeftBound11481.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70134

namespace LeftBound70139
def owner : Owner := ⟨.program ⟨214⟩, ⟨11467⟩⟩
def transferEvent : Nat := 70139
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 70137 .coefficient, .predecessor 1 70138 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70137 .coefficient)
      LeftBound70134.bound (LeftBound70134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact70136RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70138 .coefficient)
      LeftBound70129.bound (LeftBound70129.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact70131RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70129.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70129.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound70134.bound, LeftBound70129.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70134.bound, LeftBound70129.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound70134.actual selector witness, LeftBound70129.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound70139

namespace LeftBound70143
def owner : Owner := ⟨.program ⟨214⟩, ⟨11468⟩⟩
def transferEvent : Nat := 70143
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 70141 .coefficient, .predecessor 1 70142 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70141 .coefficient)
      LeftBound70139.bound (LeftBound70139.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact70140RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70139.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70139.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70142 .coefficient)
      LeftBound11473.bound (LeftBound11473.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11474RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11473.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11473.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound70139.bound, LeftBound11473.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70139.bound, LeftBound11473.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound70139.actual selector witness, LeftBound11473.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound70143

namespace LeftBound70144
def owner : Owner := ⟨.program ⟨214⟩, ⟨11468⟩⟩
def transferEvent : Nat := 70144
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨93⟩⟩]⟩ [⟨.result 11474 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11474 .coefficient)
      LeftBound11473.bound (LeftBound11473.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨93⟩⟩) (rawTerms := some (Proof.Events044.exact11474RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11473.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11473.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound11473.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11473.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound11473.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound70144

namespace LeftBound70149
def owner : Owner := ⟨.program ⟨214⟩, ⟨14201⟩⟩
def transferEvent : Nat := 70149
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 70147 .coefficient) (.predecessor 1 70148 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70147 .coefficient)
      LeftBound70143.bound (LeftBound70143.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events274.exact70146RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70143.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70143.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70148 .coefficient)
      LeftAuthority3318.bound (LeftAuthority3318.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events012.exact3319RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3318.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3318.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound70143.bound LeftAuthority3318.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70143.bound, LeftAuthority3318.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound70143.actual selector witness) * (LeftAuthority3318.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70149

namespace LeftBound70150
def owner : Owner := ⟨.program ⟨214⟩, ⟨14201⟩⟩
def transferEvent : Nat := 70150
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨14198⟩⟩], []⟩ [⟨.result 3319 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 3319 .coefficient)
      LeftAuthority3318.bound (LeftAuthority3318.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨14198⟩⟩) (rawTerms := some (Proof.Events012.exact3319RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3318.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3318.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority3318.bound []
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3318.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority3318.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound70150

namespace LeftBound70151
def owner : Owner := ⟨.program ⟨214⟩, ⟨14201⟩⟩
def transferEvent : Nat := 70151
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 70146 .summary) (.transfer 70150) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 70146 .summary)
      LeftBound70144.bound (LeftBound70144.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11468⟩⟩) (rawTerms := some (Proof.Events274.exact70146RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound70144.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 70150)
      LeftBound70150.bound (LeftBound70150.actual selector witness) := by
  exact .transfer (LeftBound70150.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound70144.bound LeftBound70150.bound
def bound : CoeffClass := .finite ⟨14976, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70144.bound, LeftBound70150.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound70144.actual selector witness) * (LeftBound70150.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70151

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
