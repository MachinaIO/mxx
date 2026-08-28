import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard604

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound88277
def owner : Owner := ⟨.program ⟨214⟩, ⟨10576⟩⟩
def transferEvent : Nat := 88277
def frameStart : Nat := 88227
def rule : BoundRule := .sum [.predecessor 0 88275 .coefficient, .predecessor 1 88276 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88275 .coefficient)
      LeftBound88260.bound (LeftBound88260.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound88260.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88276 .coefficient)
      LeftAuthority88273.bound (LeftAuthority88273.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority88273.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88260.bound, LeftAuthority88273.bound]
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88260.bound, LeftAuthority88273.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88260.actual selector witness, LeftAuthority88273.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88277

namespace LeftBound88280
def owner : Owner := ⟨.program ⟨214⟩, ⟨10577⟩⟩
def transferEvent : Nat := 88280
def frameStart : Nat := 88227
def rule : BoundRule := .identity (.predecessor 0 88279 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88279 .coefficient)
      LeftBound88277.bound (LeftBound88277.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound88277.derived selector witness)

def rawBound : CoeffClass := LeftBound88277.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88277.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound88277.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound88280

namespace LeftBound88286
def owner : Owner := ⟨.program ⟨214⟩, ⟨10578⟩⟩
def transferEvent : Nat := 88286
def frameStart : Nat := 88227
def rule : BoundRule := .product (.predecessor 0 88284 .coefficient) (.predecessor 1 88285 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88284 .coefficient)
      LeftAuthority88282.bound (LeftAuthority88282.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events344.exact88283RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority88282.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority88282.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88285 .coefficient)
      LeftBound88280.bound (LeftBound88280.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events344.exact88281RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88280.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88280.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority88282.bound LeftBound88280.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority88282.bound, LeftBound88280.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority88282.actual selector witness) * (LeftBound88280.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound88286

namespace LeftBound88300
def owner : Owner := ⟨.program ⟨214⟩, ⟨7832⟩⟩
def transferEvent : Nat := 88300
def frameStart : Nat := 88227
def rule : BoundRule := .scale (.predecessor 0 88298 .coefficient) (.value (.predecessor 1 88299 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88298 .coefficient)
      LeftAuthority88296.bound (LeftAuthority88296.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events344.exact88297RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority88296.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority88296.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88299 .coefficient)
      LeftAuthority88230.bound (LeftAuthority88230.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority88230.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority88296.bound LeftAuthority88230.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority88296.bound, LeftAuthority88230.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority88296.actual selector witness) * (LeftAuthority88230.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound88300

namespace LeftBound88303
def owner : Owner := ⟨.program ⟨214⟩, ⟨6771⟩⟩
def transferEvent : Nat := 88303
def frameStart : Nat := 88227
def rule : BoundRule := .identity (.predecessor 0 88302 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88302 .coefficient)
      LeftAuthority88290.bound (LeftAuthority88290.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events344.exact88291RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority88290.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority88290.derived selector witness)

def rawBound : CoeffClass := LeftAuthority88290.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority88290.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority88290.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound88303

namespace LeftBound88307
def owner : Owner := ⟨.program ⟨214⟩, ⟨7833⟩⟩
def transferEvent : Nat := 88307
def frameStart : Nat := 88227
def rule : BoundRule := .product (.predecessor 0 88305 .coefficient) (.predecessor 1 88306 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88305 .coefficient)
      LeftBound88303.bound (LeftBound88303.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events344.exact88304RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88303.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88303.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88306 .coefficient)
      LeftBound88300.bound (LeftBound88300.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events344.exact88301RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88300.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88300.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound88303.bound LeftBound88300.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88303.bound, LeftBound88300.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound88303.actual selector witness) * (LeftBound88300.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound88307

namespace LeftBound88312
def owner : Owner := ⟨.program ⟨214⟩, ⟨10579⟩⟩
def transferEvent : Nat := 88312
def frameStart : Nat := 88227
def rule : BoundRule := .sum [.predecessor 0 88310 .coefficient, .predecessor 1 88311 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88310 .coefficient)
      LeftBound88307.bound (LeftBound88307.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events344.exact88309RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88307.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88307.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88311 .coefficient)
      LeftBound88286.bound (LeftBound88286.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events344.exact88288RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88286.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88286.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88307.bound, LeftBound88286.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88307.bound, LeftBound88286.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88307.actual selector witness, LeftBound88286.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88312

namespace LeftBound88316
def owner : Owner := ⟨.program ⟨214⟩, ⟨24914⟩⟩
def transferEvent : Nat := 88316
def frameStart : Nat := 88227
def rule : BoundRule := .product (.predecessor 0 88314 .coefficient) (.predecessor 1 88315 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88314 .coefficient)
      LeftBound88312.bound (LeftBound88312.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events344.exact88313RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88312.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88312.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88315 .coefficient)
      LeftAuthority88271.bound (LeftAuthority88271.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events344.exact88272RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority88271.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority88271.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound88312.bound LeftAuthority88271.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88312.bound, LeftAuthority88271.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound88312.actual selector witness) * (LeftAuthority88271.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound88316

namespace LeftBound88327
def owner : Owner := ⟨.program ⟨214⟩, ⟨14794⟩⟩
def transferEvent : Nat := 88327
def frameStart : Nat := 88227
def rule : BoundRule := .product (.predecessor 0 88325 .coefficient) (.predecessor 1 88326 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88325 .coefficient)
      LeftAuthority88282.bound (LeftAuthority88282.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events344.exact88283RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority88282.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority88282.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88326 .coefficient)
      LeftAuthority88323.bound (LeftAuthority88323.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events345.exact88324RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority88323.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority88323.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority88282.bound LeftAuthority88323.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority88282.bound, LeftAuthority88323.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority88282.actual selector witness) * (LeftAuthority88323.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound88327

namespace LeftBound88335
def owner : Owner := ⟨.program ⟨214⟩, ⟨14795⟩⟩
def transferEvent : Nat := 88335
def frameStart : Nat := 88227
def rule : BoundRule := .sum [.predecessor 0 88333 .coefficient, .predecessor 1 88334 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88333 .coefficient)
      LeftAuthority88331.bound (LeftAuthority88331.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events345.exact88332RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority88331.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority88331.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88334 .coefficient)
      LeftBound88327.bound (LeftBound88327.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events345.exact88329RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88327.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88327.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority88331.bound, LeftBound88327.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority88331.bound, LeftBound88327.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority88331.actual selector witness, LeftBound88327.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88335

namespace LeftBound88339
def owner : Owner := ⟨.program ⟨214⟩, ⟨24915⟩⟩
def transferEvent : Nat := 88339
def frameStart : Nat := 88227
def rule : BoundRule := .sum [.predecessor 0 88337 .coefficient, .predecessor 1 88338 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88337 .coefficient)
      LeftBound88335.bound (LeftBound88335.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events345.exact88336RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88335.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88335.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88338 .coefficient)
      LeftBound88316.bound (LeftBound88316.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events345.exact88321RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88316.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88316.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88335.bound, LeftBound88316.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88335.bound, LeftBound88316.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88335.actual selector witness, LeftBound88316.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88339

namespace LeftBound88352
def owner : Owner := ⟨.program ⟨214⟩, ⟨24913⟩⟩
def transferEvent : Nat := 88352
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 88350 .coefficient, .predecessor 1 88351 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88350 .coefficient)
      LeftBound88175.bound (LeftBound88175.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events345.exact88349RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88175.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88175.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88351 .coefficient)
      LeftBound88158.bound (LeftBound88158.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events344.exact88165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88158.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88158.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88175.bound, LeftBound88158.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88175.bound, LeftBound88158.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88175.actual selector witness, LeftBound88158.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88352

namespace LeftBound88355
def owner : Owner := ⟨.program ⟨214⟩, ⟨24913⟩⟩
def transferEvent : Nat := 88355
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 88349 .summary, .result 88165 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 88349 .summary)
      LeftBound88177.bound (LeftBound88177.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19027⟩⟩) (rawTerms := some (Proof.Events345.exact88349RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound88177.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 88165 .summary)
      LeftBound88160.bound (LeftBound88160.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨24912⟩⟩) (rawTerms := some (Proof.Events344.exact88165RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound88160.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88177.bound, LeftBound88160.bound]
def bound : CoeffClass := .finite ⟨352011863863296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88177.bound, LeftBound88160.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88177.actual selector witness, LeftBound88160.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88355

namespace LeftBound88359
def owner : Owner := ⟨.program ⟨214⟩, ⟨26360⟩⟩
def transferEvent : Nat := 88359
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 88357 .coefficient) (.predecessor 1 88358 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88357 .coefficient)
      LeftBound88352.bound (LeftBound88352.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events345.exact88356RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88352.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88352.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88358 .coefficient)
      LeftAuthority88080.bound (LeftAuthority88080.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events344.exact88081RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority88080.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority88080.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound88352.bound LeftAuthority88080.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88352.bound, LeftAuthority88080.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound88352.actual selector witness) * (LeftAuthority88080.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound88359

namespace LeftBound88360
def owner : Owner := ⟨.program ⟨214⟩, ⟨26360⟩⟩
def transferEvent : Nat := 88360
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26358⟩⟩]⟩ [⟨.result 88081 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 88081 .coefficient)
      LeftAuthority88080.bound (LeftAuthority88080.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26358⟩⟩) (rawTerms := some (Proof.Events344.exact88081RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority88080.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority88080.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority88080.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority88080.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority88080.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound88360

namespace LeftBound88361
def owner : Owner := ⟨.program ⟨214⟩, ⟨26360⟩⟩
def transferEvent : Nat := 88361
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 88356 .summary) (.transfer 88360) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 88356 .summary)
      LeftBound88355.bound (LeftBound88355.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨24913⟩⟩) (rawTerms := some (Proof.Events345.exact88356RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound88355.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 88360)
      LeftBound88360.bound (LeftBound88360.actual selector witness) := by
  exact .transfer (LeftBound88360.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound88355.bound LeftBound88360.bound
def bound : CoeffClass := .finite ⟨1291889172568118132736, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88355.bound, LeftBound88360.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound88355.actual selector witness) * (LeftBound88360.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound88361

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
