import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard508

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound75225
def owner : Owner := ⟨.program ⟨214⟩, ⟨18335⟩⟩
def transferEvent : Nat := 75225
def frameStart : Nat := 74728
def rule : BoundRule := .sum [.predecessor 0 75223 .coefficient, .predecessor 1 75224 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75223 .coefficient)
      LeftBound75221.bound (LeftBound75221.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events293.exact75222RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75221.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75221.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75224 .coefficient)
      LeftAuthority74793.bound (LeftAuthority74793.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events292.exact74794RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority74793.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority74793.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75221.bound, LeftAuthority74793.bound]
def bound : CoeffClass := .finite ⟨996, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75221.bound, LeftAuthority74793.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound75221.actual selector witness, LeftAuthority74793.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75225

namespace LeftBound75229
def owner : Owner := ⟨.program ⟨214⟩, ⟨18336⟩⟩
def transferEvent : Nat := 75229
def frameStart : Nat := 74728
def rule : BoundRule := .sum [.predecessor 0 75227 .coefficient, .predecessor 1 75228 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75227 .coefficient)
      LeftBound75225.bound (LeftBound75225.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events293.exact75226RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75225.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75225.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75228 .coefficient)
      LeftAuthority74770.bound (LeftAuthority74770.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events292.exact74771RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority74770.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority74770.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75225.bound, LeftAuthority74770.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75225.bound, LeftAuthority74770.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound75225.actual selector witness, LeftAuthority74770.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75229

namespace LeftBound75232
def owner : Owner := ⟨.program ⟨214⟩, ⟨18337⟩⟩
def transferEvent : Nat := 75232
def frameStart : Nat := 74728
def rule : BoundRule := .identity (.predecessor 0 75231 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75231 .coefficient)
      LeftBound75229.bound (LeftBound75229.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events293.exact75230RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75229.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75229.derived selector witness)

def rawBound : CoeffClass := LeftBound75229.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75229.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound75229.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound75232

namespace LeftBound75249
def owner : Owner := ⟨.program ⟨214⟩, ⟨18643⟩⟩
def transferEvent : Nat := 75249
def frameStart : Nat := 74728
def rule : BoundRule := .sum [.predecessor 0 75247 .coefficient, .predecessor 1 75248 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75247 .coefficient)
      LeftBound75232.bound (LeftBound75232.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound75232.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75248 .coefficient)
      LeftAuthority75245.bound (LeftAuthority75245.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority75245.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75232.bound, LeftAuthority75245.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75232.bound, LeftAuthority75245.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound75232.actual selector witness, LeftAuthority75245.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75249

namespace LeftBound75252
def owner : Owner := ⟨.program ⟨214⟩, ⟨18644⟩⟩
def transferEvent : Nat := 75252
def frameStart : Nat := 74728
def rule : BoundRule := .identity (.predecessor 0 75251 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75251 .coefficient)
      LeftBound75249.bound (LeftBound75249.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound75249.derived selector witness)

def rawBound : CoeffClass := LeftBound75249.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75249.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound75249.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound75252

namespace LeftBound75258
def owner : Owner := ⟨.program ⟨214⟩, ⟨18645⟩⟩
def transferEvent : Nat := 75258
def frameStart : Nat := 74728
def rule : BoundRule := .product (.predecessor 0 75256 .coefficient) (.predecessor 1 75257 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75256 .coefficient)
      LeftAuthority75254.bound (LeftAuthority75254.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events293.exact75255RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75254.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75254.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75257 .coefficient)
      LeftBound75252.bound (LeftBound75252.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events293.exact75253RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75252.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75252.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority75254.bound LeftBound75252.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority75254.bound, LeftBound75252.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority75254.actual selector witness) * (LeftBound75252.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound75258

namespace LeftBound75334
def owner : Owner := ⟨.program ⟨214⟩, ⟨6795⟩⟩
def transferEvent : Nat := 75334
def frameStart : Nat := 74728
def rule : BoundRule := .sum [.predecessor 0 75332 .coefficient, .predecessor 1 75333 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75332 .coefficient)
      LeftAuthority75330.bound (LeftAuthority75330.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75331RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75330.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75330.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75333 .coefficient)
      LeftAuthority75327.bound (LeftAuthority75327.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75328RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75327.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75327.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority75330.bound, LeftAuthority75327.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority75330.bound, LeftAuthority75327.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority75330.actual selector witness, LeftAuthority75327.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75334

namespace LeftBound75338
def owner : Owner := ⟨.program ⟨214⟩, ⟨6796⟩⟩
def transferEvent : Nat := 75338
def frameStart : Nat := 74728
def rule : BoundRule := .sum [.predecessor 0 75336 .coefficient, .predecessor 1 75337 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75336 .coefficient)
      LeftBound75334.bound (LeftBound75334.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75335RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75334.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75334.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75337 .coefficient)
      LeftAuthority75324.bound (LeftAuthority75324.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75325RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75324.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75324.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75334.bound, LeftAuthority75324.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75334.bound, LeftAuthority75324.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound75334.actual selector witness, LeftAuthority75324.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75338

namespace LeftBound75342
def owner : Owner := ⟨.program ⟨214⟩, ⟨6797⟩⟩
def transferEvent : Nat := 75342
def frameStart : Nat := 74728
def rule : BoundRule := .sum [.predecessor 0 75340 .coefficient, .predecessor 1 75341 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75340 .coefficient)
      LeftBound75338.bound (LeftBound75338.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75339RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75338.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75338.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75341 .coefficient)
      LeftAuthority75321.bound (LeftAuthority75321.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75322RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75321.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75321.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75338.bound, LeftAuthority75321.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75338.bound, LeftAuthority75321.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound75338.actual selector witness, LeftAuthority75321.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75342

namespace LeftBound75346
def owner : Owner := ⟨.program ⟨214⟩, ⟨6798⟩⟩
def transferEvent : Nat := 75346
def frameStart : Nat := 74728
def rule : BoundRule := .sum [.predecessor 0 75344 .coefficient, .predecessor 1 75345 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75344 .coefficient)
      LeftBound75342.bound (LeftBound75342.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75343RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75342.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75342.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75345 .coefficient)
      LeftAuthority75318.bound (LeftAuthority75318.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75319RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75318.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75318.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75342.bound, LeftAuthority75318.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75342.bound, LeftAuthority75318.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound75342.actual selector witness, LeftAuthority75318.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75346

namespace LeftBound75350
def owner : Owner := ⟨.program ⟨214⟩, ⟨6799⟩⟩
def transferEvent : Nat := 75350
def frameStart : Nat := 74728
def rule : BoundRule := .sum [.predecessor 0 75348 .coefficient, .predecessor 1 75349 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75348 .coefficient)
      LeftBound75346.bound (LeftBound75346.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75347RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75346.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75346.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75349 .coefficient)
      LeftAuthority75315.bound (LeftAuthority75315.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75316RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75315.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75315.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75346.bound, LeftAuthority75315.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75346.bound, LeftAuthority75315.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound75346.actual selector witness, LeftAuthority75315.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75350

namespace LeftBound75354
def owner : Owner := ⟨.program ⟨214⟩, ⟨6800⟩⟩
def transferEvent : Nat := 75354
def frameStart : Nat := 74728
def rule : BoundRule := .sum [.predecessor 0 75352 .coefficient, .predecessor 1 75353 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75352 .coefficient)
      LeftBound75350.bound (LeftBound75350.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75351RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75350.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75350.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75353 .coefficient)
      LeftAuthority75312.bound (LeftAuthority75312.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75313RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75312.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75312.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75350.bound, LeftAuthority75312.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75350.bound, LeftAuthority75312.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound75350.actual selector witness, LeftAuthority75312.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75354

namespace LeftBound75358
def owner : Owner := ⟨.program ⟨214⟩, ⟨6801⟩⟩
def transferEvent : Nat := 75358
def frameStart : Nat := 74728
def rule : BoundRule := .sum [.predecessor 0 75356 .coefficient, .predecessor 1 75357 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75356 .coefficient)
      LeftBound75354.bound (LeftBound75354.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75355RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75354.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75354.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75357 .coefficient)
      LeftAuthority75309.bound (LeftAuthority75309.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75310RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75309.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75309.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75354.bound, LeftAuthority75309.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75354.bound, LeftAuthority75309.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound75354.actual selector witness, LeftAuthority75309.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75358

namespace LeftBound75362
def owner : Owner := ⟨.program ⟨214⟩, ⟨6802⟩⟩
def transferEvent : Nat := 75362
def frameStart : Nat := 74728
def rule : BoundRule := .sum [.predecessor 0 75360 .coefficient, .predecessor 1 75361 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75360 .coefficient)
      LeftBound75358.bound (LeftBound75358.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75359RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75358.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75358.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75361 .coefficient)
      LeftAuthority75306.bound (LeftAuthority75306.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75307RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75306.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75306.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75358.bound, LeftAuthority75306.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75358.bound, LeftAuthority75306.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound75358.actual selector witness, LeftAuthority75306.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75362

namespace LeftBound75366
def owner : Owner := ⟨.program ⟨214⟩, ⟨6803⟩⟩
def transferEvent : Nat := 75366
def frameStart : Nat := 74728
def rule : BoundRule := .sum [.predecessor 0 75364 .coefficient, .predecessor 1 75365 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75364 .coefficient)
      LeftBound75362.bound (LeftBound75362.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75363RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75362.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75362.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75365 .coefficient)
      LeftAuthority75303.bound (LeftAuthority75303.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75304RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75303.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75303.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75362.bound, LeftAuthority75303.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75362.bound, LeftAuthority75303.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound75362.actual selector witness, LeftAuthority75303.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75366

namespace LeftBound75370
def owner : Owner := ⟨.program ⟨214⟩, ⟨6804⟩⟩
def transferEvent : Nat := 75370
def frameStart : Nat := 74728
def rule : BoundRule := .sum [.predecessor 0 75368 .coefficient, .predecessor 1 75369 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75368 .coefficient)
      LeftBound75366.bound (LeftBound75366.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75367RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75366.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75366.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75369 .coefficient)
      LeftAuthority75300.bound (LeftAuthority75300.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75301RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75300.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75300.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75366.bound, LeftAuthority75300.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75366.bound, LeftAuthority75300.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound75366.actual selector witness, LeftAuthority75300.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75370

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
