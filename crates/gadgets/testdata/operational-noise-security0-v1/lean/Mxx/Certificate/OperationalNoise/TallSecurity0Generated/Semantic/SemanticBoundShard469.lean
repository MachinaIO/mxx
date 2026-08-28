import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard060
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard061
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard437
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard440
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard468

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound69121
def owner : Owner := ⟨.program ⟨214⟩, ⟨16307⟩⟩
def transferEvent : Nat := 69121
def frameStart : Nat := 69025
def rule : BoundRule := .sum [.predecessor 0 69119 .coefficient, .predecessor 1 69120 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69119 .coefficient)
      LeftAuthority69117.bound (LeftAuthority69117.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events269.exact69118RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority69117.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority69117.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69120 .coefficient)
      LeftBound69113.bound (LeftBound69113.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events269.exact69115RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69113.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69113.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority69117.bound, LeftBound69113.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority69117.bound, LeftBound69113.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority69117.actual selector witness, LeftBound69113.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69121

namespace LeftBound69125
def owner : Owner := ⟨.program ⟨214⟩, ⟨28509⟩⟩
def transferEvent : Nat := 69125
def frameStart : Nat := 69025
def rule : BoundRule := .sum [.predecessor 0 69123 .coefficient, .predecessor 1 69124 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69123 .coefficient)
      LeftBound69121.bound (LeftBound69121.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events270.exact69122RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69121.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69121.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69124 .coefficient)
      LeftBound69102.bound (LeftBound69102.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events269.exact69107RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69102.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69102.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69121.bound, LeftBound69102.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69121.bound, LeftBound69102.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound69121.actual selector witness, LeftBound69102.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69125

namespace LeftBound69138
def owner : Owner := ⟨.program ⟨214⟩, ⟨28507⟩⟩
def transferEvent : Nat := 69138
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 69136 .coefficient, .predecessor 1 69137 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69136 .coefficient)
      LeftBound68967.bound (LeftBound68967.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events270.exact69135RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68967.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68967.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69137 .coefficient)
      LeftBound68950.bound (LeftBound68950.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events269.exact68957RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68950.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68950.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound68967.bound, LeftBound68950.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68967.bound, LeftBound68950.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound68967.actual selector witness, LeftBound68950.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69138

namespace LeftBound69141
def owner : Owner := ⟨.program ⟨214⟩, ⟨28507⟩⟩
def transferEvent : Nat := 69141
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 69135 .summary, .result 68957 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 69135 .summary)
      LeftBound68969.bound (LeftBound68969.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21831⟩⟩) (rawTerms := some (Proof.Events270.exact69135RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound68969.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 68957 .summary)
      LeftBound68952.bound (LeftBound68952.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28506⟩⟩) (rawTerms := some (Proof.Events269.exact68957RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound68952.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound68969.bound, LeftBound68952.bound]
def bound : CoeffClass := .finite ⟨1292202948609709846528, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68969.bound, LeftBound68952.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound68969.actual selector witness, LeftBound68952.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69141

namespace LeftBound69165
def owner : Owner := ⟨.program ⟨214⟩, ⟨11634⟩⟩
def transferEvent : Nat := 69165
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 69163 .coefficient) (.predecessor 1 69164 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69163 .coefficient)
      LeftAuthority3269.bound (LeftAuthority3269.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events012.exact3270RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3269.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3269.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69164 .coefficient)
      LeftBound65293.bound (LeftBound65293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65293.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority3269.bound LeftBound65293.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3269.bound, LeftBound65293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority3269.actual selector witness) * (LeftBound65293.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound69165

namespace LeftBound69170
def owner : Owner := ⟨.program ⟨214⟩, ⟨7199⟩⟩
def transferEvent : Nat := 69170
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 69168 .coefficient) (.predecessor 1 69169 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69168 .coefficient)
      LeftBound65164.bound (LeftBound65164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65164.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69169 .coefficient)
      LeftBound10479.bound (LeftBound10479.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10480RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10479.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10479.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound65164.bound LeftBound10479.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65164.bound, LeftBound10479.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound65164.actual selector witness) * (LeftBound10479.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound69170

namespace LeftBound69175
def owner : Owner := ⟨.program ⟨214⟩, ⟨11635⟩⟩
def transferEvent : Nat := 69175
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 69173 .coefficient, .predecessor 1 69174 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69173 .coefficient)
      LeftBound69170.bound (LeftBound69170.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events270.exact69172RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69170.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69170.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69174 .coefficient)
      LeftBound69165.bound (LeftBound69165.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events270.exact69167RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69165.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69165.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69170.bound, LeftBound69165.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69170.bound, LeftBound69165.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound69170.actual selector witness, LeftBound69165.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69175

namespace LeftBound69179
def owner : Owner := ⟨.program ⟨214⟩, ⟨11636⟩⟩
def transferEvent : Nat := 69179
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 69177 .coefficient, .predecessor 1 69178 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69177 .coefficient)
      LeftBound69175.bound (LeftBound69175.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events270.exact69176RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69175.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69175.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69178 .coefficient)
      LeftBound10471.bound (LeftBound10471.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10472RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10471.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10471.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69175.bound, LeftBound10471.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69175.bound, LeftBound10471.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound69175.actual selector witness, LeftBound10471.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69179

namespace LeftBound69180
def owner : Owner := ⟨.program ⟨214⟩, ⟨11636⟩⟩
def transferEvent : Nat := 69180
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨95⟩⟩]⟩ [⟨.result 10472 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10472 .coefficient)
      LeftBound10471.bound (LeftBound10471.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨95⟩⟩) (rawTerms := some (Proof.Events040.exact10472RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10471.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10471.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound10471.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10471.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound10471.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound69180

namespace LeftBound69185
def owner : Owner := ⟨.program ⟨214⟩, ⟨14635⟩⟩
def transferEvent : Nat := 69185
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 69183 .coefficient) (.predecessor 1 69184 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69183 .coefficient)
      LeftBound69179.bound (LeftBound69179.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events270.exact69182RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69179.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69179.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69184 .coefficient)
      LeftAuthority3272.bound (LeftAuthority3272.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events012.exact3273RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3272.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3272.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound69179.bound LeftAuthority3272.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69179.bound, LeftAuthority3272.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound69179.actual selector witness) * (LeftAuthority3272.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound69185

namespace LeftBound69186
def owner : Owner := ⟨.program ⟨214⟩, ⟨14635⟩⟩
def transferEvent : Nat := 69186
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨14632⟩⟩], []⟩ [⟨.result 3273 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 3273 .coefficient)
      LeftAuthority3272.bound (LeftAuthority3272.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨14632⟩⟩) (rawTerms := some (Proof.Events012.exact3273RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3272.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3272.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority3272.bound []
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3272.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority3272.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound69186

namespace LeftBound69187
def owner : Owner := ⟨.program ⟨214⟩, ⟨14635⟩⟩
def transferEvent : Nat := 69187
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 69182 .summary) (.transfer 69186) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 69182 .summary)
      LeftBound69180.bound (LeftBound69180.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11636⟩⟩) (rawTerms := some (Proof.Events270.exact69182RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound69180.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 69186)
      LeftBound69186.bound (LeftBound69186.actual selector witness) := by
  exact .transfer (LeftBound69186.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound69180.bound LeftBound69186.bound
def bound : CoeffClass := .finite ⟨23296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69180.bound, LeftBound69186.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound69180.actual selector witness) * (LeftBound69186.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound69187

namespace LeftBound69193
def owner : Owner := ⟨.program ⟨214⟩, ⟨14636⟩⟩
def transferEvent : Nat := 69193
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 69191 .coefficient) (.predecessor 1 69192 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69191 .coefficient)
      LeftAuthority3272.bound (LeftAuthority3272.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events012.exact3273RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3272.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3272.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69192 .coefficient)
      LeftBound65293.bound (LeftBound65293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65293.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority3272.bound LeftBound65293.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3272.bound, LeftBound65293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority3272.actual selector witness) * (LeftBound65293.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound69193

namespace LeftBound69198
def owner : Owner := ⟨.program ⟨214⟩, ⟨7180⟩⟩
def transferEvent : Nat := 69198
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 69196 .coefficient) (.predecessor 1 69197 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69196 .coefficient)
      LeftBound65164.bound (LeftBound65164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65164.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69197 .coefficient)
      LeftBound10520.bound (LeftBound10520.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10521RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10520.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10520.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound65164.bound LeftBound10520.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65164.bound, LeftBound10520.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound65164.actual selector witness) * (LeftBound10520.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound69198

namespace LeftBound69203
def owner : Owner := ⟨.program ⟨214⟩, ⟨14637⟩⟩
def transferEvent : Nat := 69203
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 69201 .coefficient, .predecessor 1 69202 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69201 .coefficient)
      LeftBound69198.bound (LeftBound69198.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events270.exact69200RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69198.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69198.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69202 .coefficient)
      LeftBound69193.bound (LeftBound69193.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events270.exact69195RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69193.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69193.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69198.bound, LeftBound69193.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69198.bound, LeftBound69193.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound69198.actual selector witness, LeftBound69193.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69203

namespace LeftBound69207
def owner : Owner := ⟨.program ⟨214⟩, ⟨14638⟩⟩
def transferEvent : Nat := 69207
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 69205 .coefficient, .predecessor 1 69206 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69205 .coefficient)
      LeftBound69203.bound (LeftBound69203.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events270.exact69204RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69203.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69203.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69206 .coefficient)
      LeftBound10512.bound (LeftBound10512.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10513RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10512.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10512.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69203.bound, LeftBound10512.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69203.bound, LeftBound10512.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound69203.actual selector witness, LeftBound10512.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69207

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
