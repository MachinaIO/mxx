import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard372

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound55184
def owner : Owner := ⟨.program ⟨214⟩, ⟨14434⟩⟩
def transferEvent : Nat := 55184
def frameStart : Nat := 55155
def rule : BoundRule := .product (.predecessor 0 55182 .coefficient) (.predecessor 1 55183 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55182 .coefficient)
      LeftAuthority55180.bound (LeftAuthority55180.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55181RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55180.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55180.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55183 .coefficient)
      LeftAuthority55177.bound (LeftAuthority55177.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55178RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55177.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55177.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority55180.bound LeftAuthority55177.bound
def bound : CoeffClass := .finite ⟨484, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55180.bound, LeftAuthority55177.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority55180.actual selector witness) * (LeftAuthority55177.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55184

namespace LeftBound55188
def owner : Owner := ⟨.program ⟨214⟩, ⟨14435⟩⟩
def transferEvent : Nat := 55188
def frameStart : Nat := 55155
def rule : BoundRule := .identity (.predecessor 0 55187 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55187 .coefficient)
      LeftBound55184.bound (LeftBound55184.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55186RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55184.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55184.derived selector witness)

def rawBound : CoeffClass := LeftBound55184.bound
def bound : CoeffClass := .finite ⟨484, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55184.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound55184.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound55188

namespace LeftBound55205
def owner : Owner := ⟨.program ⟨214⟩, ⟨14535⟩⟩
def transferEvent : Nat := 55205
def frameStart : Nat := 55155
def rule : BoundRule := .sum [.predecessor 0 55203 .coefficient, .predecessor 1 55204 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55203 .coefficient)
      LeftBound55188.bound (LeftBound55188.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound55188.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55204 .coefficient)
      LeftAuthority55201.bound (LeftAuthority55201.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority55201.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55188.bound, LeftAuthority55201.bound]
def bound : CoeffClass := .finite ⟨484, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55188.bound, LeftAuthority55201.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound55188.actual selector witness, LeftAuthority55201.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55205

namespace LeftBound55208
def owner : Owner := ⟨.program ⟨214⟩, ⟨14536⟩⟩
def transferEvent : Nat := 55208
def frameStart : Nat := 55155
def rule : BoundRule := .identity (.predecessor 0 55207 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55207 .coefficient)
      LeftBound55205.bound (LeftBound55205.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound55205.derived selector witness)

def rawBound : CoeffClass := LeftBound55205.bound
def bound : CoeffClass := .finite ⟨484, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55205.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound55205.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound55208

namespace LeftBound55214
def owner : Owner := ⟨.program ⟨214⟩, ⟨14537⟩⟩
def transferEvent : Nat := 55214
def frameStart : Nat := 55155
def rule : BoundRule := .product (.predecessor 0 55212 .coefficient) (.predecessor 1 55213 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55212 .coefficient)
      LeftAuthority55210.bound (LeftAuthority55210.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55211RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55210.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55210.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55213 .coefficient)
      LeftBound55208.bound (LeftBound55208.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55209RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55208.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55208.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority55210.bound LeftBound55208.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55210.bound, LeftBound55208.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority55210.actual selector witness) * (LeftBound55208.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55214

namespace LeftBound55230
def owner : Owner := ⟨.program ⟨214⟩, ⟨7856⟩⟩
def transferEvent : Nat := 55230
def frameStart : Nat := 55155
def rule : BoundRule := .scale (.predecessor 0 55228 .coefficient) (.value (.predecessor 1 55229 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55228 .coefficient)
      LeftAuthority55226.bound (LeftAuthority55226.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55227RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55226.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55226.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55229 .coefficient)
      LeftAuthority55217.bound (LeftAuthority55217.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority55217.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority55226.bound LeftAuthority55217.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55226.bound, LeftAuthority55217.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority55226.actual selector witness) * (LeftAuthority55217.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound55230

namespace LeftBound55233
def owner : Owner := ⟨.program ⟨214⟩, ⟨6761⟩⟩
def transferEvent : Nat := 55233
def frameStart : Nat := 55155
def rule : BoundRule := .identity (.predecessor 0 55232 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55232 .coefficient)
      LeftAuthority55220.bound (LeftAuthority55220.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55221RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55220.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55220.derived selector witness)

def rawBound : CoeffClass := LeftAuthority55220.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55220.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority55220.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound55233

namespace LeftBound55237
def owner : Owner := ⟨.program ⟨214⟩, ⟨7857⟩⟩
def transferEvent : Nat := 55237
def frameStart : Nat := 55155
def rule : BoundRule := .product (.predecessor 0 55235 .coefficient) (.predecessor 1 55236 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55235 .coefficient)
      LeftBound55233.bound (LeftBound55233.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55234RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55233.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55233.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55236 .coefficient)
      LeftBound55230.bound (LeftBound55230.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55231RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55230.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55230.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound55233.bound LeftBound55230.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55233.bound, LeftBound55230.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound55233.actual selector witness) * (LeftBound55230.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55237

namespace LeftBound55242
def owner : Owner := ⟨.program ⟨214⟩, ⟨14538⟩⟩
def transferEvent : Nat := 55242
def frameStart : Nat := 55155
def rule : BoundRule := .sum [.predecessor 0 55240 .coefficient, .predecessor 1 55241 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55240 .coefficient)
      LeftBound55237.bound (LeftBound55237.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55239RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55237.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55237.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55241 .coefficient)
      LeftBound55214.bound (LeftBound55214.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55216RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55214.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55214.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55237.bound, LeftBound55214.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55237.bound, LeftBound55214.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound55237.actual selector witness, LeftBound55214.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55242

namespace LeftBound55246
def owner : Owner := ⟨.program ⟨214⟩, ⟨26151⟩⟩
def transferEvent : Nat := 55246
def frameStart : Nat := 55155
def rule : BoundRule := .product (.predecessor 0 55244 .coefficient) (.predecessor 1 55245 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55244 .coefficient)
      LeftBound55242.bound (LeftBound55242.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55243RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55242.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55242.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55245 .coefficient)
      LeftAuthority55199.bound (LeftAuthority55199.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55200RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55199.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55199.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound55242.bound LeftAuthority55199.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55242.bound, LeftAuthority55199.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound55242.actual selector witness) * (LeftAuthority55199.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55246

namespace LeftBound55257
def owner : Owner := ⟨.program ⟨214⟩, ⟨16065⟩⟩
def transferEvent : Nat := 55257
def frameStart : Nat := 55155
def rule : BoundRule := .product (.predecessor 0 55255 .coefficient) (.predecessor 1 55256 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55255 .coefficient)
      LeftAuthority55210.bound (LeftAuthority55210.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55211RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55210.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55210.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55256 .coefficient)
      LeftAuthority55253.bound (LeftAuthority55253.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55254RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55253.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55253.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority55210.bound LeftAuthority55253.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55210.bound, LeftAuthority55253.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority55210.actual selector witness) * (LeftAuthority55253.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55257

namespace LeftBound55265
def owner : Owner := ⟨.program ⟨214⟩, ⟨16066⟩⟩
def transferEvent : Nat := 55265
def frameStart : Nat := 55155
def rule : BoundRule := .sum [.predecessor 0 55263 .coefficient, .predecessor 1 55264 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55263 .coefficient)
      LeftAuthority55261.bound (LeftAuthority55261.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55262RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55261.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55261.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55264 .coefficient)
      LeftBound55257.bound (LeftBound55257.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55259RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55257.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55257.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority55261.bound, LeftBound55257.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55261.bound, LeftBound55257.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority55261.actual selector witness, LeftBound55257.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55265

namespace LeftBound55269
def owner : Owner := ⟨.program ⟨214⟩, ⟨26152⟩⟩
def transferEvent : Nat := 55269
def frameStart : Nat := 55155
def rule : BoundRule := .sum [.predecessor 0 55267 .coefficient, .predecessor 1 55268 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55267 .coefficient)
      LeftBound55265.bound (LeftBound55265.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55266RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55265.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55265.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55268 .coefficient)
      LeftBound55246.bound (LeftBound55246.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55251RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55246.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55246.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55265.bound, LeftBound55246.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55265.bound, LeftBound55246.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound55265.actual selector witness, LeftBound55246.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55269

namespace LeftBound55282
def owner : Owner := ⟨.program ⟨214⟩, ⟨26150⟩⟩
def transferEvent : Nat := 55282
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 55280 .coefficient, .predecessor 1 55281 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55280 .coefficient)
      LeftBound55103.bound (LeftBound55103.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55279RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55103.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55103.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55281 .coefficient)
      LeftBound55086.bound (LeftBound55086.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55093RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55086.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55086.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55103.bound, LeftBound55086.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55103.bound, LeftBound55086.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound55103.actual selector witness, LeftBound55086.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55282

namespace LeftBound55285
def owner : Owner := ⟨.program ⟨214⟩, ⟨26150⟩⟩
def transferEvent : Nat := 55285
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 55279 .summary, .result 55093 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 55279 .summary)
      LeftBound55105.bound (LeftBound55105.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19607⟩⟩) (rawTerms := some (Proof.Events215.exact55279RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound55105.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 55093 .summary)
      LeftBound55088.bound (LeftBound55088.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26149⟩⟩) (rawTerms := some (Proof.Events215.exact55093RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound55088.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55105.bound, LeftBound55088.bound]
def bound : CoeffClass := .finite ⟨352072932929536, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55105.bound, LeftBound55088.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound55105.actual selector witness, LeftBound55088.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55285

namespace LeftBound55289
def owner : Owner := ⟨.program ⟨214⟩, ⟨28098⟩⟩
def transferEvent : Nat := 55289
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 55287 .coefficient) (.predecessor 1 55288 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55287 .coefficient)
      LeftBound55282.bound (LeftBound55282.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55286RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55282.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55282.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55288 .coefficient)
      LeftAuthority55008.bound (LeftAuthority55008.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events214.exact55009RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55008.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55008.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound55282.bound LeftAuthority55008.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55282.bound, LeftAuthority55008.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound55282.actual selector witness) * (LeftAuthority55008.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55289

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
