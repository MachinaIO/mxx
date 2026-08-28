import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard492

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound72272
def owner : Owner := ⟨.program ⟨214⟩, ⟨7842⟩⟩
def transferEvent : Nat := 72272
def frameStart : Nat := 72190
def rule : BoundRule := .product (.predecessor 0 72270 .coefficient) (.predecessor 1 72271 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72270 .coefficient)
      LeftBound72268.bound (LeftBound72268.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events282.exact72269RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72268.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72268.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72271 .coefficient)
      LeftBound72265.bound (LeftBound72265.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events282.exact72266RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72265.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72265.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound72268.bound LeftBound72265.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72268.bound, LeftBound72265.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound72268.actual selector witness) * (LeftBound72265.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72272

namespace LeftBound72277
def owner : Owner := ⟨.program ⟨214⟩, ⟨12269⟩⟩
def transferEvent : Nat := 72277
def frameStart : Nat := 72190
def rule : BoundRule := .sum [.predecessor 0 72275 .coefficient, .predecessor 1 72276 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72275 .coefficient)
      LeftBound72272.bound (LeftBound72272.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events282.exact72274RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72272.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72272.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72276 .coefficient)
      LeftBound72249.bound (LeftBound72249.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events282.exact72251RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72249.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72249.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound72272.bound, LeftBound72249.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72272.bound, LeftBound72249.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound72272.actual selector witness, LeftBound72249.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72277

namespace LeftBound72281
def owner : Owner := ⟨.program ⟨214⟩, ⟨25294⟩⟩
def transferEvent : Nat := 72281
def frameStart : Nat := 72190
def rule : BoundRule := .product (.predecessor 0 72279 .coefficient) (.predecessor 1 72280 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72279 .coefficient)
      LeftBound72277.bound (LeftBound72277.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events282.exact72278RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72277.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72277.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72280 .coefficient)
      LeftAuthority72234.bound (LeftAuthority72234.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events282.exact72235RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72234.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72234.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound72277.bound LeftAuthority72234.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72277.bound, LeftAuthority72234.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound72277.actual selector witness) * (LeftAuthority72234.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72281

namespace LeftBound72292
def owner : Owner := ⟨.program ⟨214⟩, ⟨15420⟩⟩
def transferEvent : Nat := 72292
def frameStart : Nat := 72190
def rule : BoundRule := .product (.predecessor 0 72290 .coefficient) (.predecessor 1 72291 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72290 .coefficient)
      LeftAuthority72245.bound (LeftAuthority72245.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events282.exact72246RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72245.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72245.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72291 .coefficient)
      LeftAuthority72288.bound (LeftAuthority72288.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events282.exact72289RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72288.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72288.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority72245.bound LeftAuthority72288.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority72245.bound, LeftAuthority72288.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority72245.actual selector witness) * (LeftAuthority72288.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72292

namespace LeftBound72300
def owner : Owner := ⟨.program ⟨214⟩, ⟨15421⟩⟩
def transferEvent : Nat := 72300
def frameStart : Nat := 72190
def rule : BoundRule := .sum [.predecessor 0 72298 .coefficient, .predecessor 1 72299 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72298 .coefficient)
      LeftAuthority72296.bound (LeftAuthority72296.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events282.exact72297RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72296.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72296.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72299 .coefficient)
      LeftBound72292.bound (LeftBound72292.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events282.exact72294RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72292.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72292.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority72296.bound, LeftBound72292.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority72296.bound, LeftBound72292.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority72296.actual selector witness, LeftBound72292.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72300

namespace LeftBound72304
def owner : Owner := ⟨.program ⟨214⟩, ⟨25295⟩⟩
def transferEvent : Nat := 72304
def frameStart : Nat := 72190
def rule : BoundRule := .sum [.predecessor 0 72302 .coefficient, .predecessor 1 72303 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72302 .coefficient)
      LeftBound72300.bound (LeftBound72300.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events282.exact72301RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72300.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72300.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72303 .coefficient)
      LeftBound72281.bound (LeftBound72281.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events282.exact72286RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72281.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72281.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound72300.bound, LeftBound72281.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72300.bound, LeftBound72281.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound72300.actual selector witness, LeftBound72281.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72304

namespace LeftBound72317
def owner : Owner := ⟨.program ⟨214⟩, ⟨25293⟩⟩
def transferEvent : Nat := 72317
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 72315 .coefficient, .predecessor 1 72316 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72315 .coefficient)
      LeftBound72138.bound (LeftBound72138.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events282.exact72314RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72138.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72138.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72316 .coefficient)
      LeftBound72121.bound (LeftBound72121.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact72128RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72121.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72121.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound72138.bound, LeftBound72121.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72138.bound, LeftBound72121.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound72138.actual selector witness, LeftBound72121.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72317

namespace LeftBound72320
def owner : Owner := ⟨.program ⟨214⟩, ⟨25293⟩⟩
def transferEvent : Nat := 72320
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 72314 .summary, .result 72128 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 72314 .summary)
      LeftBound72140.bound (LeftBound72140.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19239⟩⟩) (rawTerms := some (Proof.Events282.exact72314RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound72140.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 72128 .summary)
      LeftBound72123.bound (LeftBound72123.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25292⟩⟩) (rawTerms := some (Proof.Events281.exact72128RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound72123.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound72140.bound, LeftBound72123.bound]
def bound : CoeffClass := .finite ⟨352024077676544, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72140.bound, LeftBound72123.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound72140.actual selector witness, LeftBound72123.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72320

namespace LeftBound72324
def owner : Owner := ⟨.program ⟨214⟩, ⟨26987⟩⟩
def transferEvent : Nat := 72324
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 72322 .coefficient) (.predecessor 1 72323 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72322 .coefficient)
      LeftBound72317.bound (LeftBound72317.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events282.exact72321RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72317.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72317.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72323 .coefficient)
      LeftAuthority72043.bound (LeftAuthority72043.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact72044RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72043.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72043.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound72317.bound LeftAuthority72043.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72317.bound, LeftAuthority72043.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound72317.actual selector witness) * (LeftAuthority72043.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72324

namespace LeftBound72325
def owner : Owner := ⟨.program ⟨214⟩, ⟨26987⟩⟩
def transferEvent : Nat := 72325
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26985⟩⟩]⟩ [⟨.result 72044 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 72044 .coefficient)
      LeftAuthority72043.bound (LeftAuthority72043.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26985⟩⟩) (rawTerms := some (Proof.Events281.exact72044RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72043.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72043.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority72043.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority72043.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority72043.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound72325

namespace LeftBound72326
def owner : Owner := ⟨.program ⟨214⟩, ⟨26987⟩⟩
def transferEvent : Nat := 72326
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 72321 .summary) (.transfer 72325) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 72321 .summary)
      LeftBound72320.bound (LeftBound72320.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25293⟩⟩) (rawTerms := some (Proof.Events282.exact72321RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound72320.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 72325)
      LeftBound72325.bound (LeftBound72325.actual selector witness) := by
  exact .transfer (LeftBound72325.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound72320.bound LeftBound72325.bound
def bound : CoeffClass := .finite ⟨1291933997458159304704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72320.bound, LeftBound72325.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound72320.actual selector witness) * (LeftBound72325.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72326

namespace LeftBound72337
def owner : Owner := ⟨.program ⟨214⟩, ⟨20822⟩⟩
def transferEvent : Nat := 72337
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 72335 .coefficient) (.value (.predecessor 1 72336 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72335 .coefficient)
      LeftAuthority72333.bound (LeftAuthority72333.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events282.exact72334RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72333.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72333.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72336 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority72333.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority72333.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority72333.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound72337

namespace LeftBound72341
def owner : Owner := ⟨.program ⟨214⟩, ⟨20823⟩⟩
def transferEvent : Nat := 72341
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 72339 .coefficient) (.predecessor 1 72340 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72339 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72340 .coefficient)
      LeftBound72337.bound (LeftBound72337.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events282.exact72338RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72337.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72337.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound72337.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound72337.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound72337.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72341

namespace LeftBound72342
def owner : Owner := ⟨.program ⟨214⟩, ⟨20823⟩⟩
def transferEvent : Nat := 72342
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20820⟩⟩]⟩ [⟨.result 72334 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 72334 .coefficient)
      LeftAuthority72333.bound (LeftAuthority72333.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20820⟩⟩) (rawTerms := some (Proof.Events282.exact72334RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72333.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72333.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority72333.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority72333.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority72333.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound72342

namespace LeftBound72343
def owner : Owner := ⟨.program ⟨214⟩, ⟨20823⟩⟩
def transferEvent : Nat := 72343
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 72342) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 72342)
      LeftBound72342.bound (LeftBound72342.actual selector witness) := by
  exact .transfer (LeftBound72342.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound72342.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound72342.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound72342.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72343

namespace LeftBound72438
def owner : Owner := ⟨.program ⟨214⟩, ⟨15419⟩⟩
def transferEvent : Nat := 72438
def frameStart : Nat := 72399
def rule : BoundRule := .identity (.predecessor 0 72437 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72437 .coefficient)
      LeftAuthority72435.bound (LeftAuthority72435.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events282.exact72436RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72435.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72435.derived selector witness)

def rawBound : CoeffClass := LeftAuthority72435.bound
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority72435.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority72435.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound72438

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
