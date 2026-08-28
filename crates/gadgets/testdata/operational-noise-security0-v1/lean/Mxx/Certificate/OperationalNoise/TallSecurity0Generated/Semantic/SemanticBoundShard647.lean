import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard036
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard646

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound95185
def owner : Owner := ⟨.program ⟨214⟩, ⟨16962⟩⟩
def transferEvent : Nat := 95185
def frameStart : Nat := 95124
def rule : BoundRule := .sum [.predecessor 0 95183 .coefficient, .predecessor 1 95184 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95183 .coefficient)
      LeftAuthority95181.bound (LeftAuthority95181.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events371.exact95182RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95181.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95181.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95184 .coefficient)
      LeftBound95177.bound (LeftBound95177.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events371.exact95179RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95177.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95177.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority95181.bound, LeftBound95177.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority95181.bound, LeftBound95177.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority95181.actual selector witness, LeftBound95177.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound95185

namespace LeftBound95189
def owner : Owner := ⟨.program ⟨214⟩, ⟨29785⟩⟩
def transferEvent : Nat := 95189
def frameStart : Nat := 95124
def rule : BoundRule := .product (.predecessor 0 95187 .coefficient) (.predecessor 1 95188 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95187 .coefficient)
      LeftBound95185.bound (LeftBound95185.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events371.exact95186RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95185.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95185.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95188 .coefficient)
      LeftAuthority95162.bound (LeftAuthority95162.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events371.exact95163RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95162.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95162.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound95185.bound LeftAuthority95162.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95185.bound, LeftAuthority95162.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound95185.actual selector witness) * (LeftAuthority95162.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95189

namespace LeftBound95200
def owner : Owner := ⟨.program ⟨214⟩, ⟨17079⟩⟩
def transferEvent : Nat := 95200
def frameStart : Nat := 95124
def rule : BoundRule := .product (.predecessor 0 95198 .coefficient) (.predecessor 1 95199 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95198 .coefficient)
      LeftAuthority95173.bound (LeftAuthority95173.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events371.exact95174RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95173.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95173.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95199 .coefficient)
      LeftAuthority95196.bound (LeftAuthority95196.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events371.exact95197RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95196.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95196.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority95173.bound LeftAuthority95196.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority95173.bound, LeftAuthority95196.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority95173.actual selector witness) * (LeftAuthority95196.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95200

namespace LeftBound95208
def owner : Owner := ⟨.program ⟨214⟩, ⟨17080⟩⟩
def transferEvent : Nat := 95208
def frameStart : Nat := 95124
def rule : BoundRule := .sum [.predecessor 0 95206 .coefficient, .predecessor 1 95207 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95206 .coefficient)
      LeftAuthority95204.bound (LeftAuthority95204.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events371.exact95205RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95204.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95204.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95207 .coefficient)
      LeftBound95200.bound (LeftBound95200.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events371.exact95202RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95200.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95200.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority95204.bound, LeftBound95200.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority95204.bound, LeftBound95200.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority95204.actual selector witness, LeftBound95200.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound95208

namespace LeftBound95212
def owner : Owner := ⟨.program ⟨214⟩, ⟨29789⟩⟩
def transferEvent : Nat := 95212
def frameStart : Nat := 95124
def rule : BoundRule := .sum [.predecessor 0 95210 .coefficient, .predecessor 1 95211 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95210 .coefficient)
      LeftBound95208.bound (LeftBound95208.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events371.exact95209RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95208.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95208.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95211 .coefficient)
      LeftBound95189.bound (LeftBound95189.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events371.exact95194RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95189.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95189.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound95208.bound, LeftBound95189.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95208.bound, LeftBound95189.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound95208.actual selector witness, LeftBound95189.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound95212

namespace LeftBound95225
def owner : Owner := ⟨.program ⟨214⟩, ⟨29787⟩⟩
def transferEvent : Nat := 95225
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 95223 .coefficient, .predecessor 1 95224 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95223 .coefficient)
      LeftBound95078.bound (LeftBound95078.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events371.exact95222RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95078.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95078.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95224 .coefficient)
      LeftBound95061.bound (LeftBound95061.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events371.exact95068RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95061.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95061.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound95078.bound, LeftBound95061.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95078.bound, LeftBound95061.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound95078.actual selector witness, LeftBound95061.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound95225

namespace LeftBound95228
def owner : Owner := ⟨.program ⟨214⟩, ⟨29787⟩⟩
def transferEvent : Nat := 95228
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 95222 .summary, .result 95068 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 95222 .summary)
      LeftBound95080.bound (LeftBound95080.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22688⟩⟩) (rawTerms := some (Proof.Events371.exact95222RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound95080.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 95068 .summary)
      LeftBound95063.bound (LeftBound95063.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29786⟩⟩) (rawTerms := some (Proof.Events371.exact95068RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound95063.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound95080.bound, LeftBound95063.bound]
def bound : CoeffClass := .finite ⟨1292516722839998050304, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95080.bound, LeftBound95063.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound95080.actual selector witness, LeftBound95063.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound95228

namespace LeftBound95252
def owner : Owner := ⟨.program ⟨214⟩, ⟨12937⟩⟩
def transferEvent : Nat := 95252
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 95250 .coefficient) (.predecessor 1 95251 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95250 .coefficient)
      LeftAuthority4611.bound (LeftAuthority4611.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events018.exact4612RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4611.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4611.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95251 .coefficient)
      LeftBound30.bound (LeftBound30.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact32RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority4611.bound LeftBound30.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4611.bound, LeftBound30.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority4611.actual selector witness) * (LeftBound30.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound95252

namespace LeftBound95257
def owner : Owner := ⟨.program ⟨214⟩, ⟨7125⟩⟩
def transferEvent : Nat := 95257
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 95255 .coefficient) (.predecessor 1 95256 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95255 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95256 .coefficient)
      LeftBound7473.bound (LeftBound7473.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7474RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7473.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7473.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound26.bound LeftBound7473.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftBound7473.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound26.actual selector witness) * (LeftBound7473.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95257

namespace LeftBound95262
def owner : Owner := ⟨.program ⟨214⟩, ⟨12938⟩⟩
def transferEvent : Nat := 95262
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 95260 .coefficient, .predecessor 1 95261 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95260 .coefficient)
      LeftBound95257.bound (LeftBound95257.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events372.exact95259RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95257.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95257.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95261 .coefficient)
      LeftBound95252.bound (LeftBound95252.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events372.exact95254RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95252.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95252.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound95257.bound, LeftBound95252.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95257.bound, LeftBound95252.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound95257.actual selector witness, LeftBound95252.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound95262

namespace LeftBound95266
def owner : Owner := ⟨.program ⟨214⟩, ⟨12939⟩⟩
def transferEvent : Nat := 95266
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 95264 .coefficient, .predecessor 1 95265 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95264 .coefficient)
      LeftBound95262.bound (LeftBound95262.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events372.exact95263RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95262.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95262.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95265 .coefficient)
      LeftBound7465.bound (LeftBound7465.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7466RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7465.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7465.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound95262.bound, LeftBound7465.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95262.bound, LeftBound7465.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound95262.actual selector witness, LeftBound7465.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound95266

namespace LeftBound95267
def owner : Owner := ⟨.program ⟨214⟩, ⟨12939⟩⟩
def transferEvent : Nat := 95267
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨102⟩⟩]⟩ [⟨.result 7466 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7466 .coefficient)
      LeftBound7465.bound (LeftBound7465.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨102⟩⟩) (rawTerms := some (Proof.Events029.exact7466RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7465.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7465.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound7465.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7465.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound7465.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound95267

namespace LeftBound95272
def owner : Owner := ⟨.program ⟨214⟩, ⟨12940⟩⟩
def transferEvent : Nat := 95272
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 95270 .coefficient) (.predecessor 1 95271 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95270 .coefficient)
      LeftBound95266.bound (LeftBound95266.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events372.exact95269RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95266.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95266.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95271 .coefficient)
      LeftAuthority4614.bound (LeftAuthority4614.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events018.exact4615RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4614.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4614.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound95266.bound LeftAuthority4614.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95266.bound, LeftAuthority4614.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound95266.actual selector witness) * (LeftAuthority4614.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95272

namespace LeftBound95273
def owner : Owner := ⟨.program ⟨214⟩, ⟨12940⟩⟩
def transferEvent : Nat := 95273
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨10120⟩⟩], []⟩ [⟨.result 4615 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 4615 .coefficient)
      LeftAuthority4614.bound (LeftAuthority4614.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨10120⟩⟩) (rawTerms := some (Proof.Events018.exact4615RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4614.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4614.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority4614.bound []
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4614.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority4614.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound95273

namespace LeftBound95274
def owner : Owner := ⟨.program ⟨214⟩, ⟨12940⟩⟩
def transferEvent : Nat := 95274
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 95269 .summary) (.transfer 95273) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 95269 .summary)
      LeftBound95267.bound (LeftBound95267.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12939⟩⟩) (rawTerms := some (Proof.Events372.exact95269RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound95267.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 95273)
      LeftBound95273.bound (LeftBound95273.actual selector witness) := by
  exact .transfer (LeftBound95273.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound95267.bound LeftBound95273.bound
def bound : CoeffClass := .finite ⟨43264, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95267.bound, LeftBound95273.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound95267.actual selector witness) * (LeftBound95273.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95274

namespace LeftBound95280
def owner : Owner := ⟨.program ⟨214⟩, ⟨10121⟩⟩
def transferEvent : Nat := 95280
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 95278 .coefficient) (.predecessor 1 95279 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95278 .coefficient)
      LeftAuthority4614.bound (LeftAuthority4614.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events018.exact4615RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4614.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4614.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95279 .coefficient)
      LeftBound30.bound (LeftBound30.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact32RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority4614.bound LeftBound30.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4614.bound, LeftBound30.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority4614.actual selector witness) * (LeftBound30.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound95280

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
