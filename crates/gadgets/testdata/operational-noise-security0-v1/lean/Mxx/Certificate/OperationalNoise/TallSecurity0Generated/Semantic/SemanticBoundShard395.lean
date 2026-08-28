import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard394

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound58106
def owner : Owner := ⟨.program ⟨214⟩, ⟨11079⟩⟩
def transferEvent : Nat := 58106
def frameStart : Nat := 58047
def rule : BoundRule := .product (.predecessor 0 58104 .coefficient) (.predecessor 1 58105 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58104 .coefficient)
      LeftAuthority58102.bound (LeftAuthority58102.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events226.exact58103RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority58102.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority58102.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58105 .coefficient)
      LeftBound58100.bound (LeftBound58100.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events226.exact58101RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58100.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58100.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority58102.bound LeftBound58100.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority58102.bound, LeftBound58100.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority58102.actual selector witness) * (LeftBound58100.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound58106

namespace LeftBound58122
def owner : Owner := ⟨.program ⟨214⟩, ⟨7838⟩⟩
def transferEvent : Nat := 58122
def frameStart : Nat := 58047
def rule : BoundRule := .scale (.predecessor 0 58120 .coefficient) (.value (.predecessor 1 58121 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58120 .coefficient)
      LeftAuthority58118.bound (LeftAuthority58118.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events227.exact58119RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority58118.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority58118.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58121 .coefficient)
      LeftAuthority58109.bound (LeftAuthority58109.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority58109.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority58118.bound LeftAuthority58109.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority58118.bound, LeftAuthority58109.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority58118.actual selector witness) * (LeftAuthority58109.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound58122

namespace LeftBound58125
def owner : Owner := ⟨.program ⟨214⟩, ⟨6791⟩⟩
def transferEvent : Nat := 58125
def frameStart : Nat := 58047
def rule : BoundRule := .identity (.predecessor 0 58124 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58124 .coefficient)
      LeftAuthority58112.bound (LeftAuthority58112.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events227.exact58113RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority58112.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority58112.derived selector witness)

def rawBound : CoeffClass := LeftAuthority58112.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority58112.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority58112.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound58125

namespace LeftBound58129
def owner : Owner := ⟨.program ⟨214⟩, ⟨7839⟩⟩
def transferEvent : Nat := 58129
def frameStart : Nat := 58047
def rule : BoundRule := .product (.predecessor 0 58127 .coefficient) (.predecessor 1 58128 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58127 .coefficient)
      LeftBound58125.bound (LeftBound58125.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events227.exact58126RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58125.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58125.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58128 .coefficient)
      LeftBound58122.bound (LeftBound58122.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events227.exact58123RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58122.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58122.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound58125.bound LeftBound58122.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58125.bound, LeftBound58122.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound58125.actual selector witness) * (LeftBound58122.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound58129

namespace LeftBound58134
def owner : Owner := ⟨.program ⟨214⟩, ⟨11080⟩⟩
def transferEvent : Nat := 58134
def frameStart : Nat := 58047
def rule : BoundRule := .sum [.predecessor 0 58132 .coefficient, .predecessor 1 58133 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58132 .coefficient)
      LeftBound58129.bound (LeftBound58129.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events227.exact58131RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58129.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58129.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58133 .coefficient)
      LeftBound58106.bound (LeftBound58106.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events226.exact58108RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58106.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58106.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound58129.bound, LeftBound58106.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58129.bound, LeftBound58106.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound58129.actual selector witness, LeftBound58106.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound58134

namespace LeftBound58138
def owner : Owner := ⟨.program ⟨214⟩, ⟨25073⟩⟩
def transferEvent : Nat := 58138
def frameStart : Nat := 58047
def rule : BoundRule := .product (.predecessor 0 58136 .coefficient) (.predecessor 1 58137 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58136 .coefficient)
      LeftBound58134.bound (LeftBound58134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events227.exact58135RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58137 .coefficient)
      LeftAuthority58091.bound (LeftAuthority58091.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events226.exact58092RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority58091.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority58091.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound58134.bound LeftAuthority58091.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58134.bound, LeftAuthority58091.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound58134.actual selector witness) * (LeftAuthority58091.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound58138

namespace LeftBound58149
def owner : Owner := ⟨.program ⟨214⟩, ⟨15120⟩⟩
def transferEvent : Nat := 58149
def frameStart : Nat := 58047
def rule : BoundRule := .product (.predecessor 0 58147 .coefficient) (.predecessor 1 58148 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58147 .coefficient)
      LeftAuthority58102.bound (LeftAuthority58102.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events226.exact58103RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority58102.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority58102.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58148 .coefficient)
      LeftAuthority58145.bound (LeftAuthority58145.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events227.exact58146RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority58145.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority58145.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority58102.bound LeftAuthority58145.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority58102.bound, LeftAuthority58145.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority58102.actual selector witness) * (LeftAuthority58145.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound58149

namespace LeftBound58157
def owner : Owner := ⟨.program ⟨214⟩, ⟨15121⟩⟩
def transferEvent : Nat := 58157
def frameStart : Nat := 58047
def rule : BoundRule := .sum [.predecessor 0 58155 .coefficient, .predecessor 1 58156 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58155 .coefficient)
      LeftAuthority58153.bound (LeftAuthority58153.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events227.exact58154RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority58153.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority58153.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58156 .coefficient)
      LeftBound58149.bound (LeftBound58149.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events227.exact58151RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58149.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58149.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority58153.bound, LeftBound58149.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority58153.bound, LeftBound58149.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority58153.actual selector witness, LeftBound58149.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound58157

namespace LeftBound58161
def owner : Owner := ⟨.program ⟨214⟩, ⟨25074⟩⟩
def transferEvent : Nat := 58161
def frameStart : Nat := 58047
def rule : BoundRule := .sum [.predecessor 0 58159 .coefficient, .predecessor 1 58160 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58159 .coefficient)
      LeftBound58157.bound (LeftBound58157.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events227.exact58158RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58157.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58157.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58160 .coefficient)
      LeftBound58138.bound (LeftBound58138.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events227.exact58143RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58138.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58138.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound58157.bound, LeftBound58138.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58157.bound, LeftBound58138.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound58157.actual selector witness, LeftBound58138.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound58161

namespace LeftBound58174
def owner : Owner := ⟨.program ⟨214⟩, ⟨25072⟩⟩
def transferEvent : Nat := 58174
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 58172 .coefficient, .predecessor 1 58173 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58172 .coefficient)
      LeftBound57995.bound (LeftBound57995.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events227.exact58171RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57995.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57995.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58173 .coefficient)
      LeftBound57978.bound (LeftBound57978.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events226.exact57985RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57978.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57978.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound57995.bound, LeftBound57978.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57995.bound, LeftBound57978.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound57995.actual selector witness, LeftBound57978.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound58174

namespace LeftBound58177
def owner : Owner := ⟨.program ⟨214⟩, ⟨25072⟩⟩
def transferEvent : Nat := 58177
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 58171 .summary, .result 57985 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 58171 .summary)
      LeftBound57997.bound (LeftBound57997.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19175⟩⟩) (rawTerms := some (Proof.Events227.exact58171RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound57997.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 57985 .summary)
      LeftBound57980.bound (LeftBound57980.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25071⟩⟩) (rawTerms := some (Proof.Events226.exact57985RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound57980.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound57997.bound, LeftBound57980.bound]
def bound : CoeffClass := .finite ⟨352017970769920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57997.bound, LeftBound57980.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound57997.actual selector witness, LeftBound57980.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound58177

namespace LeftBound58181
def owner : Owner := ⟨.program ⟨214⟩, ⟨26796⟩⟩
def transferEvent : Nat := 58181
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 58179 .coefficient) (.predecessor 1 58180 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58179 .coefficient)
      LeftBound58174.bound (LeftBound58174.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events227.exact58178RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58174.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58174.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58180 .coefficient)
      LeftAuthority57900.bound (LeftAuthority57900.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events226.exact57901RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57900.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57900.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound58174.bound LeftAuthority57900.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58174.bound, LeftAuthority57900.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound58174.actual selector witness) * (LeftAuthority57900.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound58181

namespace LeftBound58182
def owner : Owner := ⟨.program ⟨214⟩, ⟨26796⟩⟩
def transferEvent : Nat := 58182
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26794⟩⟩]⟩ [⟨.result 57901 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 57901 .coefficient)
      LeftAuthority57900.bound (LeftAuthority57900.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26794⟩⟩) (rawTerms := some (Proof.Events226.exact57901RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57900.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57900.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority57900.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority57900.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority57900.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound58182

namespace LeftBound58183
def owner : Owner := ⟨.program ⟨214⟩, ⟨26796⟩⟩
def transferEvent : Nat := 58183
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 58178 .summary) (.transfer 58182) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 58178 .summary)
      LeftBound58177.bound (LeftBound58177.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25072⟩⟩) (rawTerms := some (Proof.Events227.exact58178RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound58177.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 58182)
      LeftBound58182.bound (LeftBound58182.actual selector witness) := by
  exact .transfer (LeftBound58182.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound58177.bound LeftBound58182.bound
def bound : CoeffClass := .finite ⟨1291911585013138718720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58177.bound, LeftBound58182.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound58177.actual selector witness) * (LeftBound58182.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound58183

namespace LeftBound58194
def owner : Owner := ⟨.program ⟨214⟩, ⟨20686⟩⟩
def transferEvent : Nat := 58194
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 58192 .coefficient) (.value (.predecessor 1 58193 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58192 .coefficient)
      LeftAuthority58190.bound (LeftAuthority58190.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events227.exact58191RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority58190.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority58190.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58193 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority58190.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority58190.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority58190.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound58194

namespace LeftBound58198
def owner : Owner := ⟨.program ⟨214⟩, ⟨20687⟩⟩
def transferEvent : Nat := 58198
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 58196 .coefficient) (.predecessor 1 58197 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58196 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58197 .coefficient)
      LeftBound58194.bound (LeftBound58194.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events227.exact58195RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58194.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58194.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound58194.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound58194.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound58194.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound58198

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
