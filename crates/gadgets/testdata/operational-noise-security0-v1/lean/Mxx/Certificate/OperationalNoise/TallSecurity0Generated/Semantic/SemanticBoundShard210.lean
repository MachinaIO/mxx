import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard018
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard149
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard209

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound32291
def owner : Owner := ⟨.program ⟨214⟩, ⟨16841⟩⟩
def transferEvent : Nat := 32291
def frameStart : Nat := 32226
def rule : BoundRule := .product (.predecessor 0 32289 .coefficient) (.predecessor 1 32290 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32289 .coefficient)
      LeftAuthority32287.bound (LeftAuthority32287.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events126.exact32288RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32287.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32287.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32290 .coefficient)
      LeftBound32285.bound (LeftBound32285.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events126.exact32286RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32285.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32285.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority32287.bound LeftBound32285.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority32287.bound, LeftBound32285.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority32287.actual selector witness) * (LeftBound32285.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound32291

namespace LeftBound32299
def owner : Owner := ⟨.program ⟨214⟩, ⟨16842⟩⟩
def transferEvent : Nat := 32299
def frameStart : Nat := 32226
def rule : BoundRule := .sum [.predecessor 0 32297 .coefficient, .predecessor 1 32298 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32297 .coefficient)
      LeftAuthority32295.bound (LeftAuthority32295.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events126.exact32296RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32295.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32295.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32298 .coefficient)
      LeftBound32291.bound (LeftBound32291.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events126.exact32293RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32291.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32291.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority32295.bound, LeftBound32291.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority32295.bound, LeftBound32291.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority32295.actual selector witness, LeftBound32291.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound32299

namespace LeftBound32303
def owner : Owner := ⟨.program ⟨214⟩, ⟨29635⟩⟩
def transferEvent : Nat := 32303
def frameStart : Nat := 32226
def rule : BoundRule := .product (.predecessor 0 32301 .coefficient) (.predecessor 1 32302 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32301 .coefficient)
      LeftBound32299.bound (LeftBound32299.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events126.exact32300RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32299.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32299.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32302 .coefficient)
      LeftAuthority32276.bound (LeftAuthority32276.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events126.exact32277RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32276.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32276.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound32299.bound LeftAuthority32276.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32299.bound, LeftAuthority32276.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound32299.actual selector witness) * (LeftAuthority32276.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound32303

namespace LeftBound32314
def owner : Owner := ⟨.program ⟨214⟩, ⟨17508⟩⟩
def transferEvent : Nat := 32314
def frameStart : Nat := 32226
def rule : BoundRule := .product (.predecessor 0 32312 .coefficient) (.predecessor 1 32313 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32312 .coefficient)
      LeftAuthority32287.bound (LeftAuthority32287.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events126.exact32288RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32287.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32287.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32313 .coefficient)
      LeftAuthority32310.bound (LeftAuthority32310.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events126.exact32311RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32310.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32310.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority32287.bound LeftAuthority32310.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority32287.bound, LeftAuthority32310.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority32287.actual selector witness) * (LeftAuthority32310.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound32314

namespace LeftBound32322
def owner : Owner := ⟨.program ⟨214⟩, ⟨17509⟩⟩
def transferEvent : Nat := 32322
def frameStart : Nat := 32226
def rule : BoundRule := .sum [.predecessor 0 32320 .coefficient, .predecessor 1 32321 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32320 .coefficient)
      LeftAuthority32318.bound (LeftAuthority32318.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events126.exact32319RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32318.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32318.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32321 .coefficient)
      LeftBound32314.bound (LeftBound32314.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events126.exact32316RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32314.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32314.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority32318.bound, LeftBound32314.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority32318.bound, LeftBound32314.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority32318.actual selector witness, LeftBound32314.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound32322

namespace LeftBound32326
def owner : Owner := ⟨.program ⟨214⟩, ⟨29640⟩⟩
def transferEvent : Nat := 32326
def frameStart : Nat := 32226
def rule : BoundRule := .sum [.predecessor 0 32324 .coefficient, .predecessor 1 32325 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32324 .coefficient)
      LeftBound32322.bound (LeftBound32322.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events126.exact32323RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32322.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32322.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32325 .coefficient)
      LeftBound32303.bound (LeftBound32303.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events126.exact32308RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32303.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32303.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound32322.bound, LeftBound32303.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32322.bound, LeftBound32303.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound32322.actual selector witness, LeftBound32303.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound32326

namespace LeftBound32339
def owner : Owner := ⟨.program ⟨214⟩, ⟨29637⟩⟩
def transferEvent : Nat := 32339
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 32337 .coefficient, .predecessor 1 32338 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32337 .coefficient)
      LeftBound32168.bound (LeftBound32168.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events126.exact32336RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32168.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32168.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32338 .coefficient)
      LeftBound32151.bound (LeftBound32151.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events125.exact32158RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32151.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32151.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound32168.bound, LeftBound32151.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32168.bound, LeftBound32151.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound32168.actual selector witness, LeftBound32151.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound32339

namespace LeftBound32342
def owner : Owner := ⟨.program ⟨214⟩, ⟨29637⟩⟩
def transferEvent : Nat := 32342
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 32336 .summary, .result 32158 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 32336 .summary)
      LeftBound32170.bound (LeftBound32170.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22495⟩⟩) (rawTerms := some (Proof.Events126.exact32336RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound32170.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 32158 .summary)
      LeftBound32153.bound (LeftBound32153.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29636⟩⟩) (rawTerms := some (Proof.Events125.exact32158RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound32153.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound32170.bound, LeftBound32153.bound]
def bound : CoeffClass := .finite ⟨1292449485504936292352, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32170.bound, LeftBound32153.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound32170.actual selector witness, LeftBound32153.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound32342

namespace LeftBound32346
def owner : Owner := ⟨.program ⟨214⟩, ⟨29638⟩⟩
def transferEvent : Nat := 32346
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 32344 .coefficient) (.predecessor 1 32345 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32344 .coefficient)
      LeftBound32339.bound (LeftBound32339.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events126.exact32343RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32339.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32339.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32345 .coefficient)
      LeftBound5558.bound (LeftBound5558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5559RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5558.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound32339.bound LeftBound5558.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32339.bound, LeftBound5558.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound32339.actual selector witness) * (LeftBound5558.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound32346

namespace LeftBound32347
def owner : Owner := ⟨.program ⟨214⟩, ⟨29638⟩⟩
def transferEvent : Nat := 32347
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6661⟩⟩]⟩ [⟨.result 5555 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5555 .coefficient)
      LeftAuthority5554.bound (LeftAuthority5554.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6661⟩⟩) (rawTerms := some (Proof.Events021.exact5555RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5554.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5554.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5554.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5554.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5554.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound32347

namespace LeftBound32348
def owner : Owner := ⟨.program ⟨214⟩, ⟨29638⟩⟩
def transferEvent : Nat := 32348
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 32343 .summary) (.transfer 32347) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 32343 .summary)
      LeftBound32342.bound (LeftBound32342.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29637⟩⟩) (rawTerms := some (Proof.Events126.exact32343RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound32342.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 32347)
      LeftBound32347.bound (LeftBound32347.actual selector witness) := by
  exact .transfer (LeftBound32347.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound32342.bound LeftBound32347.bound
def bound : CoeffClass := .finite ⟨4743310290994884271912517632, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32342.bound, LeftBound32347.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound32342.actual selector witness) * (LeftBound32347.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound32348

namespace LeftBound32363
def owner : Owner := ⟨.program ⟨214⟩, ⟨29419⟩⟩
def transferEvent : Nat := 32363
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 32361 .coefficient) (.predecessor 1 32362 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32361 .coefficient)
      LeftBound23140.bound (LeftBound23140.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23144RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23140.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23140.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32362 .coefficient)
      LeftAuthority32359.bound (LeftAuthority32359.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events126.exact32360RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32359.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32359.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound23140.bound LeftAuthority32359.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23140.bound, LeftAuthority32359.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound23140.actual selector witness) * (LeftAuthority32359.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound32363

namespace LeftBound32364
def owner : Owner := ⟨.program ⟨214⟩, ⟨29419⟩⟩
def transferEvent : Nat := 32364
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29417⟩⟩]⟩ [⟨.result 32360 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 32360 .coefficient)
      LeftAuthority32359.bound (LeftAuthority32359.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29417⟩⟩) (rawTerms := some (Proof.Events126.exact32360RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32359.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32359.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority32359.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority32359.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority32359.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound32364

namespace LeftBound32365
def owner : Owner := ⟨.program ⟨214⟩, ⟨29419⟩⟩
def transferEvent : Nat := 32365
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 23144 .summary) (.transfer 32364) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 23144 .summary)
      LeftBound23143.bound (LeftBound23143.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25544⟩⟩) (rawTerms := some (Proof.Events090.exact23144RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound23143.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 32364)
      LeftBound32364.bound (LeftBound32364.actual selector witness) := by
  exact .transfer (LeftBound32364.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound23143.bound LeftBound32364.bound
def bound : CoeffClass := .finite ⟨1292382246358571024384, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23143.bound, LeftBound32364.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound23143.actual selector witness) * (LeftBound32364.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound32365

namespace LeftBound32376
def owner : Owner := ⟨.program ⟨214⟩, ⟨22350⟩⟩
def transferEvent : Nat := 32376
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 32374 .coefficient) (.value (.predecessor 1 32375 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32374 .coefficient)
      LeftAuthority32372.bound (LeftAuthority32372.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events126.exact32373RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32372.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32372.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32375 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority32372.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority32372.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority32372.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound32376

namespace LeftBound32380
def owner : Owner := ⟨.program ⟨214⟩, ⟨22351⟩⟩
def transferEvent : Nat := 32380
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 32378 .coefficient) (.predecessor 1 32379 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32378 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32379 .coefficient)
      LeftBound32376.bound (LeftBound32376.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events126.exact32377RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32376.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32376.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound32376.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound32376.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound32376.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound32380

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
