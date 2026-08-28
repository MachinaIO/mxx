import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound36225
def owner : Owner := ⟨.program ⟨214⟩, ⟨13368⟩⟩
def transferEvent : Nat := 36225
def frameStart : Nat := 36192
def rule : BoundRule := .identity (.predecessor 0 36224 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36224 .coefficient)
      LeftBound36221.bound (LeftBound36221.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36223RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36221.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36221.derived selector witness)

def rawBound : CoeffClass := LeftBound36221.bound
def bound : CoeffClass := .finite ⟨3600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36221.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound36221.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound36225

namespace LeftBound36242
def owner : Owner := ⟨.program ⟨214⟩, ⟨13454⟩⟩
def transferEvent : Nat := 36242
def frameStart : Nat := 36192
def rule : BoundRule := .sum [.predecessor 0 36240 .coefficient, .predecessor 1 36241 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36240 .coefficient)
      LeftBound36225.bound (LeftBound36225.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound36225.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36241 .coefficient)
      LeftAuthority36238.bound (LeftAuthority36238.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority36238.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound36225.bound, LeftAuthority36238.bound]
def bound : CoeffClass := .finite ⟨3600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36225.bound, LeftAuthority36238.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound36225.actual selector witness, LeftAuthority36238.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound36242

namespace LeftBound36245
def owner : Owner := ⟨.program ⟨214⟩, ⟨13455⟩⟩
def transferEvent : Nat := 36245
def frameStart : Nat := 36192
def rule : BoundRule := .identity (.predecessor 0 36244 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36244 .coefficient)
      LeftBound36242.bound (LeftBound36242.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound36242.derived selector witness)

def rawBound : CoeffClass := LeftBound36242.bound
def bound : CoeffClass := .finite ⟨3600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36242.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound36242.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound36245

namespace LeftBound36251
def owner : Owner := ⟨.program ⟨214⟩, ⟨13456⟩⟩
def transferEvent : Nat := 36251
def frameStart : Nat := 36192
def rule : BoundRule := .product (.predecessor 0 36249 .coefficient) (.predecessor 1 36250 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36249 .coefficient)
      LeftAuthority36247.bound (LeftAuthority36247.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36248RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36247.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36247.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36250 .coefficient)
      LeftBound36245.bound (LeftBound36245.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36246RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36245.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36245.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority36247.bound LeftBound36245.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority36247.bound, LeftBound36245.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority36247.actual selector witness) * (LeftBound36245.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36251

namespace LeftBound36267
def owner : Owner := ⟨.program ⟨214⟩, ⟨7883⟩⟩
def transferEvent : Nat := 36267
def frameStart : Nat := 36192
def rule : BoundRule := .scale (.predecessor 0 36265 .coefficient) (.value (.predecessor 1 36266 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36265 .coefficient)
      LeftAuthority36263.bound (LeftAuthority36263.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36264RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36263.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36263.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36266 .coefficient)
      LeftAuthority36254.bound (LeftAuthority36254.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority36254.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority36263.bound LeftAuthority36254.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority36263.bound, LeftAuthority36254.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority36263.actual selector witness) * (LeftAuthority36254.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound36267

namespace LeftBound36270
def owner : Owner := ⟨.program ⟨214⟩, ⟨6770⟩⟩
def transferEvent : Nat := 36270
def frameStart : Nat := 36192
def rule : BoundRule := .identity (.predecessor 0 36269 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36269 .coefficient)
      LeftAuthority36257.bound (LeftAuthority36257.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36258RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36257.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36257.derived selector witness)

def rawBound : CoeffClass := LeftAuthority36257.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority36257.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority36257.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound36270

namespace LeftBound36274
def owner : Owner := ⟨.program ⟨214⟩, ⟨7884⟩⟩
def transferEvent : Nat := 36274
def frameStart : Nat := 36192
def rule : BoundRule := .product (.predecessor 0 36272 .coefficient) (.predecessor 1 36273 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36272 .coefficient)
      LeftBound36270.bound (LeftBound36270.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36271RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36270.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36270.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36273 .coefficient)
      LeftBound36267.bound (LeftBound36267.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36268RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36267.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36267.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound36270.bound LeftBound36267.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36270.bound, LeftBound36267.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound36270.actual selector witness) * (LeftBound36267.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36274

namespace LeftBound36279
def owner : Owner := ⟨.program ⟨214⟩, ⟨13457⟩⟩
def transferEvent : Nat := 36279
def frameStart : Nat := 36192
def rule : BoundRule := .sum [.predecessor 0 36277 .coefficient, .predecessor 1 36278 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36277 .coefficient)
      LeftBound36274.bound (LeftBound36274.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36276RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36274.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36274.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36278 .coefficient)
      LeftBound36251.bound (LeftBound36251.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36253RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36251.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36251.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound36274.bound, LeftBound36251.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36274.bound, LeftBound36251.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound36274.actual selector witness, LeftBound36251.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound36279

namespace LeftBound36283
def owner : Owner := ⟨.program ⟨214⟩, ⟨25771⟩⟩
def transferEvent : Nat := 36283
def frameStart : Nat := 36192
def rule : BoundRule := .product (.predecessor 0 36281 .coefficient) (.predecessor 1 36282 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36281 .coefficient)
      LeftBound36279.bound (LeftBound36279.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36280RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36279.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36279.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36282 .coefficient)
      LeftAuthority36236.bound (LeftAuthority36236.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36237RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36236.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36236.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound36279.bound LeftAuthority36236.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36279.bound, LeftAuthority36236.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound36279.actual selector witness) * (LeftAuthority36236.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36283

namespace LeftBound36294
def owner : Owner := ⟨.program ⟨214⟩, ⟨17021⟩⟩
def transferEvent : Nat := 36294
def frameStart : Nat := 36192
def rule : BoundRule := .product (.predecessor 0 36292 .coefficient) (.predecessor 1 36293 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36292 .coefficient)
      LeftAuthority36247.bound (LeftAuthority36247.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36248RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36247.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36247.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36293 .coefficient)
      LeftAuthority36290.bound (LeftAuthority36290.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36291RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36290.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36290.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority36247.bound LeftAuthority36290.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority36247.bound, LeftAuthority36290.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority36247.actual selector witness) * (LeftAuthority36290.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36294

namespace LeftBound36302
def owner : Owner := ⟨.program ⟨214⟩, ⟨17022⟩⟩
def transferEvent : Nat := 36302
def frameStart : Nat := 36192
def rule : BoundRule := .sum [.predecessor 0 36300 .coefficient, .predecessor 1 36301 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36300 .coefficient)
      LeftAuthority36298.bound (LeftAuthority36298.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36299RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36298.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36298.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36301 .coefficient)
      LeftBound36294.bound (LeftBound36294.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36296RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36294.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36294.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority36298.bound, LeftBound36294.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority36298.bound, LeftBound36294.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority36298.actual selector witness, LeftBound36294.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound36302

namespace LeftBound36306
def owner : Owner := ⟨.program ⟨214⟩, ⟨25772⟩⟩
def transferEvent : Nat := 36306
def frameStart : Nat := 36192
def rule : BoundRule := .sum [.predecessor 0 36304 .coefficient, .predecessor 1 36305 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36304 .coefficient)
      LeftBound36302.bound (LeftBound36302.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36303RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36302.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36302.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36305 .coefficient)
      LeftBound36283.bound (LeftBound36283.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36288RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36283.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36283.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound36302.bound, LeftBound36283.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36302.bound, LeftBound36283.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound36302.actual selector witness, LeftBound36283.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound36306

namespace LeftBound36319
def owner : Owner := ⟨.program ⟨214⟩, ⟨25770⟩⟩
def transferEvent : Nat := 36319
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 36317 .coefficient, .predecessor 1 36318 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36317 .coefficient)
      LeftBound36140.bound (LeftBound36140.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36316RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36140.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36140.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36318 .coefficient)
      LeftBound36112.bound (LeftBound36112.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36119RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36112.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36112.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound36140.bound, LeftBound36112.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36140.bound, LeftBound36112.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound36140.actual selector witness, LeftBound36112.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound36319

namespace LeftBound36322
def owner : Owner := ⟨.program ⟨214⟩, ⟨25770⟩⟩
def transferEvent : Nat := 36322
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 36316 .summary, .result 36119 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36316 .summary)
      LeftBound36142.bound (LeftBound36142.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20259⟩⟩) (rawTerms := some (Proof.Events141.exact36316RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36142.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36119 .summary)
      LeftBound36114.bound (LeftBound36114.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25769⟩⟩) (rawTerms := some (Proof.Events141.exact36119RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36114.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound36142.bound, LeftBound36114.bound]
def bound : CoeffClass := .finite ⟨352188964155392, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36142.bound, LeftBound36114.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound36142.actual selector witness, LeftBound36114.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound36322

namespace LeftBound36326
def owner : Owner := ⟨.program ⟨214⟩, ⟨30163⟩⟩
def transferEvent : Nat := 36326
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 36324 .coefficient) (.predecessor 1 36325 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36324 .coefficient)
      LeftBound36319.bound (LeftBound36319.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36323RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36319.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36319.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36325 .coefficient)
      LeftAuthority36029.bound (LeftAuthority36029.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact36030RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36029.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36029.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound36319.bound LeftAuthority36029.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36319.bound, LeftAuthority36029.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound36319.actual selector witness) * (LeftAuthority36029.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36326

namespace LeftBound36327
def owner : Owner := ⟨.program ⟨214⟩, ⟨30163⟩⟩
def transferEvent : Nat := 36327
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨30161⟩⟩]⟩ [⟨.result 36030 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36030 .coefficient)
      LeftAuthority36029.bound (LeftAuthority36029.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨30161⟩⟩) (rawTerms := some (Proof.Events140.exact36030RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36029.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36029.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority36029.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority36029.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority36029.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound36327

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
